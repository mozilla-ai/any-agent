import { defineConfig } from "astro/config";
import starlight from "@astrojs/starlight";

const site = process.env.CI
  ? "https://mozilla-ai.github.io"
  : "http://localhost:4321";

export default defineConfig({
  site,
  base: "/any-agent",
  integrations: [
    starlight({
      title: "any-agent",
      logo: {
        src: "./src/assets/any-agent-logo-mark.png",
      },
      favicon: "/images/any-agent_favicon.png",
      social: [
        {
          icon: "github",
          label: "GitHub",
          href: "https://github.com/mozilla-ai/any-agent",
        },
      ],
      customCss: ["./src/styles/custom.css"],
      editLink: {
        baseUrl: "https://github.com/mozilla-ai/any-agent/edit/main/docs/",
      },
      sidebar: [
        { label: "Introduction", link: "/" },
        {
          label: "Agents",
          items: [
            { label: "Defining and Running Agents", slug: "agents" },
            { label: "Models", slug: "agents/models" },
            { label: "Callbacks", slug: "agents/callbacks" },
            {
              label: "Frameworks",
              items: [
                { label: "Agno", slug: "agents/frameworks/agno" },
                { label: "Google ADK", slug: "agents/frameworks/google-adk" },
                { label: "LangChain", slug: "agents/frameworks/langchain" },
                { label: "LlamaIndex", slug: "agents/frameworks/llama-index" },
                {
                  label: "OpenAI Agents SDK",
                  slug: "agents/frameworks/openai",
                },
                { label: "smolagents", slug: "agents/frameworks/smolagents" },
                { label: "TinyAgent", slug: "agents/frameworks/tinyagent" },
              ],
            },
            { label: "Tools", slug: "agents/tools" },
          ],
        },
        { label: "Frameworks", slug: "frameworks" },
        { label: "Tracing", slug: "tracing" },
        { label: "Evaluation", slug: "evaluation" },
        { label: "Serving", slug: "serving" },
        {
          label: "Cookbook",
          items: [
            { label: "Your First Agent", slug: "cookbook/your-first-agent" },
            {
              label: "Your First Agent Evaluation",
              slug: "cookbook/your-first-agent-evaluation",
            },
            { label: "Using Callbacks", slug: "cookbook/callbacks" },
            { label: "MCP Agent", slug: "cookbook/mcp-agent" },
            { label: "Serve with A2A", slug: "cookbook/serve-a2a" },
            {
              label: "Use an Agent as a Tool (A2A)",
              slug: "cookbook/a2a-as-tool",
            },
            { label: "Local Agent", slug: "cookbook/agent-with-local-llm" },
          ],
        },
        {
          label: "API Reference",
          items: [
            { label: "Agent", slug: "api/agent" },
            { label: "Callbacks", slug: "api/callbacks" },
            { label: "Config", slug: "api/config" },
            { label: "Evaluation", slug: "api/evaluation" },
            { label: "Logging", slug: "api/logging" },
            { label: "Serving", slug: "api/serving" },
            { label: "Tools", slug: "api/tools" },
            { label: "Tracing", slug: "api/tracing" },
          ],
        },
      ],
      head: [
        {
          tag: "script",
          attrs: { type: "application/ld+json" },
          content: JSON.stringify({
            "@context": "https://schema.org",
            "@type": "SoftwareSourceCode",
            name: "any-agent",
            description:
              "A Python library providing a single interface to different agent frameworks",
            programmingLanguage: "Python",
            codeRepository: "https://github.com/mozilla-ai/any-agent",
            license:
              "https://github.com/mozilla-ai/any-agent/blob/main/LICENSE",
            author: {
              "@type": "Organization",
              name: "Mozilla.ai",
              url: "https://mozilla.ai",
            },
          }),
        },
      ],
    }),
  ],
});
