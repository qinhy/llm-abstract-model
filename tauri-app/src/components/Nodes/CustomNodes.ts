// file: CodeBlockExtract.ts
import { defineNode, NodeInterface, TextInputInterface, TextareaInputInterface, TextInterface, NumberInterface, IntegerInterface } from "baklavajs";

export const PromptOutput = defineNode({
    type: "PromptOutput",
    title: "Display",
    inputs: {
      source: () => new NodeInterface("Source", null)
    },
    outputs: {
      output: () => new TextInterface("Display", "null")
    },
    calculate(inputs) {
      console.log(inputs);
      
      let code;
      code = inputs.source ? inputs.source : "-1";
      return {
        output: code
      };
    }
  });  

export const PromptInput = defineNode({
    type: "PromptInput",
    inputs: {
        source: () => new TextareaInputInterface("Text", "Hi!").setPort(false),
    },
    outputs: {
        output: () => new NodeInterface<string>("Output", "null"),
    },
    calculate(inputs) {
        return {output:inputs.source};
    },
});

export const CodeBlockExtract = defineNode({
    type: "CodeBlockExtract",
    inputs: {
        source: () => new TextInputInterface("Text", ""),
        header: () => new TextInputInterface("Text", "python").setPort(false),
    },
    outputs: {
        output: () => new NodeInterface<string>("Output", "null"),
    },
    calculate(inputs) {
        return {output:inputs.header};
    },
});

export const TimeSleepNode = defineNode({
  type: "TimeSleepNode",
  title: "Time Sleep",
  inputs: {
    delay: () => new IntegerInterface("Delay (ms)", 1000), // Default delay set to 1000ms
  },
  outputs: {
    done: () => new IntegerInterface("Done", 0).setPort(false) // Output trigger
  },
  async calculate({delay}) {
    console.log("Starting sleep for:", delay, "ms");
    // Simulate delay with a promise
    const d = delay ?? 1000; // Default to 1000ms if no input provided
    await new Promise((resolve) => setTimeout(resolve, d));
    console.log("Sleep complete");
    // Trigger the done output
    return {done:1};
  }
});