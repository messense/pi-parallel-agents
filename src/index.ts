/**
 * Pi Parallel Agents Extension
 *
 * A dynamic parallel execution system that allows multiple agents to run in parallel
 * with different models, without requiring pre-defined agent configurations.
 *
 * Supports four execution modes:
 * - Single: One task with optional model/tools override
 * - Parallel: Multiple tasks running concurrently
 * - Chain: Sequential execution with {previous} placeholder
 * - Race: Multiple models compete on the same task
 * 
 * Agents can be specified inline (model, tools, systemPrompt) or by referencing
 * existing agent definitions from ~/.pi/agent/agents or .pi/agents.
 */

import type { ExtensionAPI } from "@mariozechner/pi-coding-agent";
import {
  type ParallelParams,
  type ParallelToolDetails,
  type TaskProgress,
  type TaskResult,
  type AgentScope,
  ParallelParamsSchema,
  createEmptyUsage,
  addUsage,
  DEFAULT_CONCURRENCY,
  MAX_CONCURRENCY,
} from "./types.js";
import { runAgent, type ExecutorOptions } from "./executor.js";
import { mapWithConcurrencyLimit, raceWithAbort } from "./parallel.js";
import { renderCall, renderResult } from "./render.js";
import { discoverAgents, findAgent, formatAgentList, type AgentConfig } from "./agents.js";

/**
 * Generate a unique task ID.
 */
function generateTaskId(index: number, name?: string): string {
  const suffix = name ? name.replace(/[^\w-]/g, "_").slice(0, 20) : `task_${index}`;
  return `${index}-${suffix}`;
}

/**
 * Aggregate usage stats from multiple results.
 */
function aggregateUsage(results: TaskResult[]): ReturnType<typeof createEmptyUsage> {
  const total = createEmptyUsage();
  for (const r of results) {
    addUsage(total, r.usage);
  }
  return total;
}

/**
 * Resolve agent settings, merging agent defaults with inline overrides.
 * Inline parameters take precedence over agent defaults.
 */
function resolveAgentSettings(
  agentName: string | undefined,
  agents: AgentConfig[],
  overrides: {
    model?: string;
    tools?: string[];
    systemPrompt?: string;
    thinking?: number | string;
  }
): {
  model?: string;
  tools?: string[];
  systemPrompt?: string;
  thinking?: number | string;
  agentConfig?: AgentConfig;
} {
  if (!agentName) {
    return { ...overrides };
  }

  const agentConfig = findAgent(agents, agentName);
  if (!agentConfig) {
    // Agent not found - just use overrides
    return { ...overrides };
  }

  // Merge: inline overrides take precedence
  return {
    model: overrides.model ?? agentConfig.model,
    tools: overrides.tools ?? agentConfig.tools,
    systemPrompt: overrides.systemPrompt ?? agentConfig.systemPrompt,
    thinking: overrides.thinking ?? agentConfig.thinking,
    agentConfig,
  };
}

export default function (pi: ExtensionAPI) {
  pi.registerTool({
    name: "parallel",
    label: "Parallel Agents",
    description: [
      "Run agents in parallel with different models.",
      "Modes:",
      "- Single: { task, model?, tools? }",
      "- Parallel: { tasks: [{task, model?, name?}, ...], maxConcurrency? }",
      "- Chain: { chain: [{task, model?}, ...] } - sequential with {previous} placeholder",
      "- Race: { race: {task, models: [...]} } - first to complete wins",
      "Model examples: claude-haiku-4-5, gpt-4o-mini, claude-sonnet-4-5",
      "",
      "Agents: Reference existing agents by name (from ~/.pi/agent/agents or .pi/agents).",
      "Set agentScope to 'both' to include project-local agents.",
      "Agent settings (model, tools, systemPrompt) are used as defaults; inline params override.",
      "",
      "IMPORTANT: Each agent runs in the same working directory with full tool access (read, bash, etc).",
      "Do NOT pre-fetch data or write temp files for agents - they can use git, grep, find, etc. directly.",
      "Just describe what they should do; they'll gather the information themselves.",
    ].join("\n"),
    parameters: ParallelParamsSchema,

    async execute(_toolCallId, params: ParallelParams, signal, onUpdate, ctx) {
      const startTime = Date.now();
      const cwd = params.cwd || ctx.cwd;
      
      // Discover available agents
      const agentScope: AgentScope = params.agentScope ?? "user";
      const discovery = discoverAgents(cwd, agentScope);
      const agents = discovery.agents;

      // Determine mode - check for non-empty values
      const hasChain = Array.isArray(params.chain) && params.chain.length > 0;
      const hasRace = params.race !== undefined && params.race !== null;
      const hasTasks = Array.isArray(params.tasks) && params.tasks.length > 0;
      const hasSingle = typeof params.task === "string" && params.task.trim().length > 0;

      const modeCount =
        Number(hasChain) + Number(hasRace) + Number(hasTasks) + Number(hasSingle);

      if (modeCount !== 1) {
        const { text: agentList } = formatAgentList(agents, 5);
        return {
          content: [
            {
              type: "text",
              text: `Invalid parameters. Provide exactly one mode: task (single), tasks (parallel), chain, or race.\n\nAvailable agents [${agentScope}]: ${agentList}`,
            },
          ],
          details: {
            mode: "single",
            results: [],
            totalDurationMs: 0,
            usage: createEmptyUsage(),
          } as ParallelToolDetails,
        };
      }

      // Helper to create details
      const makeDetails = (
        mode: ParallelToolDetails["mode"],
        results: TaskResult[],
        progress?: TaskProgress[],
        winner?: string
      ): ParallelToolDetails => ({
        mode,
        results,
        totalDurationMs: Date.now() - startTime,
        usage: aggregateUsage(results),
        progress,
        winner,
      });

      // Helper to emit progress update
      const emitUpdate = (
        mode: ParallelToolDetails["mode"],
        results: TaskResult[],
        progress: TaskProgress[],
        winner?: string
      ) => {
        onUpdate?.({
          content: [{ type: "text", text: "Running..." }],
          details: makeDetails(mode, results, progress, winner),
        });
      };

      // ========================================================================
      // Single Mode
      // ========================================================================
      if (hasSingle && params.task) {
        const progress: TaskProgress[] = [];
        
        // Resolve agent settings
        const resolved = resolveAgentSettings(params.agent, agents, {
          model: params.model,
          tools: params.tools,
          systemPrompt: params.systemPrompt,
          thinking: params.thinking,
        });
        
        // Warn if agent was specified but not found
        if (params.agent && !resolved.agentConfig) {
          const { text: agentList } = formatAgentList(agents, 5);
          return {
            content: [
              {
                type: "text",
                text: `Unknown agent: ${params.agent}\n\nAvailable agents [${agentScope}]: ${agentList}`,
              },
            ],
            details: makeDetails("single", []),
          };
        }

        const result = await runAgent({
          task: params.task,
          cwd,
          model: resolved.model,
          tools: resolved.tools,
          systemPrompt: resolved.systemPrompt,
          thinking: resolved.thinking,
          id: "single",
          name: resolved.agentConfig?.name || "single",
          signal,
          onProgress: (p) => {
            progress[0] = p;
            emitUpdate("single", [], progress);
          },
        });

        return {
          content: [{ type: "text", text: result.output || "(no output)" }],
          details: makeDetails("single", [result]),
        };
      }

      // ========================================================================
      // Chain Mode
      // ========================================================================
      if (hasChain && params.chain) {
        const results: TaskResult[] = [];
        const progress: TaskProgress[] = [];
        let previousOutput = "";
        
        // Validate all agents exist before starting
        const missingAgents: string[] = [];
        for (const step of params.chain) {
          if (step.agent && !findAgent(agents, step.agent)) {
            missingAgents.push(step.agent);
          }
        }
        if (missingAgents.length > 0) {
          const { text: agentList } = formatAgentList(agents, 5);
          return {
            content: [
              {
                type: "text",
                text: `Unknown agent(s) in chain: ${missingAgents.join(", ")}\n\nAvailable agents [${agentScope}]: ${agentList}`,
              },
            ],
            details: makeDetails("chain", []),
          };
        }

        for (let i = 0; i < params.chain.length; i++) {
          const step = params.chain[i];
          const taskWithContext = step.task.replace(/\{previous\}/g, previousOutput);
          const stepId = generateTaskId(i + 1, `step_${i + 1}`);
          
          // Resolve agent settings for this step
          const resolved = resolveAgentSettings(step.agent, agents, {
            model: step.model,
            tools: step.tools,
            systemPrompt: step.systemPrompt,
            thinking: step.thinking,
          });
          const stepName = resolved.agentConfig?.name || `Step ${i + 1}`;

          // Initialize progress for this step
          progress[i] = {
            id: stepId,
            name: stepName,
            status: "running",
            task: taskWithContext,
            model: resolved.model,
            recentTools: [],
            recentOutput: [],
            toolCount: 0,
            tokens: 0,
            durationMs: 0,
          };

          const result = await runAgent({
            task: taskWithContext,
            cwd,
            model: resolved.model,
            tools: resolved.tools,
            systemPrompt: resolved.systemPrompt,
            thinking: resolved.thinking,
            id: stepId,
            name: stepName,
            step: i + 1,
            signal,
            onProgress: (p) => {
              progress[i] = p;
              emitUpdate("chain", results, progress);
            },
          });

          results.push(result);

          // Check for failure
          if (result.exitCode !== 0 || result.aborted) {
            const errorMsg = result.error || result.output || "(no output)";
            return {
              content: [
                {
                  type: "text",
                  text: `Chain stopped at step ${i + 1}: ${errorMsg}`,
                },
              ],
              details: makeDetails("chain", results),
              isError: true,
            };
          }

          previousOutput = result.output;
        }

        const lastResult = results[results.length - 1];
        return {
          content: [
            { type: "text", text: lastResult?.output || "(no output)" },
          ],
          details: makeDetails("chain", results),
        };
      }

      // ========================================================================
      // Race Mode
      // ========================================================================
      if (hasRace && params.race) {
        const { task, models, tools, systemPrompt, thinking } = params.race;
        const results: TaskResult[] = [];
        const progress: TaskProgress[] = [];

        // Initialize progress for all racers
        for (let i = 0; i < models.length; i++) {
          const model = models[i];
          progress[i] = {
            id: model,
            name: model,
            status: "pending",
            task,
            model,
            recentTools: [],
            recentOutput: [],
            toolCount: 0,
            tokens: 0,
            durationMs: 0,
          };
        }

        const raceTasks = models.map((model, index) => ({
          id: model,
          run: async (raceSignal: AbortSignal): Promise<TaskResult> => {
            progress[index].status = "running";
            emitUpdate("race", results, progress);

            const result = await runAgent({
              task,
              cwd,
              model,
              tools,
              systemPrompt,
              thinking,
              id: model,
              name: model,
              signal: raceSignal,
              onProgress: (p) => {
                progress[index] = p;
                emitUpdate("race", results, progress);
              },
            });

            progress[index].status = result.exitCode === 0 ? "completed" : "failed";
            results.push(result);
            emitUpdate("race", results, progress);

            if (result.exitCode !== 0) {
              throw new Error(result.error || "Task failed");
            }

            return result;
          },
        }));

        const raceResult = await raceWithAbort(raceTasks, signal);

        if ("aborted" in raceResult) {
          return {
            content: [{ type: "text", text: "Race aborted" }],
            details: makeDetails("race", results, progress),
            isError: true,
          };
        }

        return {
          content: [
            { type: "text", text: raceResult.result.output || "(no output)" },
          ],
          details: makeDetails("race", results, progress, raceResult.winner),
        };
      }

      // ========================================================================
      // Parallel Mode
      // ========================================================================
      if (hasTasks && params.tasks) {
        const tasks = params.tasks;
        const maxConcurrency = Math.min(
          params.maxConcurrency || DEFAULT_CONCURRENCY,
          MAX_CONCURRENCY,
          tasks.length
        );
        
        // Validate all agents exist before starting
        const missingAgents: string[] = [];
        for (const t of tasks) {
          if (t.agent && !findAgent(agents, t.agent)) {
            missingAgents.push(t.agent);
          }
        }
        if (missingAgents.length > 0) {
          const { text: agentList } = formatAgentList(agents, 5);
          return {
            content: [
              {
                type: "text",
                text: `Unknown agent(s): ${[...new Set(missingAgents)].join(", ")}\n\nAvailable agents [${agentScope}]: ${agentList}`,
              },
            ],
            details: makeDetails("parallel", []),
          };
        }

        const progress: TaskProgress[] = tasks.map((t, i) => {
          const resolved = resolveAgentSettings(t.agent, agents, { model: t.model });
          return {
            id: generateTaskId(i, t.name || resolved.agentConfig?.name),
            name: t.name || resolved.agentConfig?.name,
            status: "pending" as const,
            task: t.task,
            model: resolved.model,
            recentTools: [],
            recentOutput: [],
            toolCount: 0,
            tokens: 0,
            durationMs: 0,
          };
        });

        const allResults: TaskResult[] = [];

        const { results: parallelResults, aborted } = await mapWithConcurrencyLimit(
          tasks,
          maxConcurrency,
          async (t, index) => {
            // Resolve agent settings for this task
            const resolved = resolveAgentSettings(t.agent, agents, {
              model: t.model,
              tools: t.tools,
              systemPrompt: t.systemPrompt,
              thinking: t.thinking,
            });
            
            const taskId = generateTaskId(index, t.name || resolved.agentConfig?.name);
            const taskName = t.name || resolved.agentConfig?.name;

            progress[index].status = "running";
            emitUpdate("parallel", allResults, progress);

            const result = await runAgent({
              task: t.task,
              cwd: t.cwd || cwd,
              model: resolved.model,
              tools: resolved.tools,
              systemPrompt: resolved.systemPrompt,
              thinking: resolved.thinking,
              context: params.context,
              id: taskId,
              name: taskName,
              signal,
              onProgress: (p) => {
                progress[index] = p;
                emitUpdate("parallel", allResults, progress);
              },
            });

            progress[index].status =
              result.exitCode === 0 ? "completed" : "failed";
            allResults.push(result);
            emitUpdate("parallel", allResults, progress);

            return result;
          },
          signal
        );

        // Collect results (filter undefined from aborted tasks)
        const results = parallelResults.filter((r): r is TaskResult => r !== undefined);

        const successCount = results.filter((r) => r.exitCode === 0).length;
        
        // Build detailed summaries - include substantial output to avoid re-running
        const summaries = results.map((r) => {
          const output = r.output.trim();
          const stats: string[] = [];
          if (r.usage.turns > 0) stats.push(`${r.usage.turns} turns`);
          if (r.model) stats.push(r.model);
          const statsStr = stats.length > 0 ? ` (${stats.join(", ")})` : "";
          const status = r.exitCode === 0 ? "✓" : "✗";
          
          // Include more output - up to 2000 chars per task
          const maxLen = 2000;
          const outputPreview = output.length > maxLen 
            ? output.slice(0, maxLen) + `\n... [${output.length - maxLen} more chars]`
            : output;
          
          return `### ${status} ${r.name || r.id}${statsStr}\n\n${outputPreview || "(no output)"}`;
        });

        return {
          content: [
            {
              type: "text",
              text: `## Parallel: ${successCount}/${results.length} succeeded${aborted ? " (aborted)" : ""}\n\n${summaries.join("\n\n---\n\n")}`,
            },
          ],
          details: makeDetails("parallel", results, progress),
        };
      }

      // Should not reach here
      return {
        content: [{ type: "text", text: "Invalid parameters" }],
        details: makeDetails("single", []),
      };
    },

    renderCall,
    renderResult,
  });
}
