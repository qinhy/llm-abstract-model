// file: CodeBlockExtract.ts
import { defineNode, NodeInterface, TextInputInterface, TextareaInputInterface, TextInterface } from "baklavajs";

const PromptOutput = defineNode({
    type: "PromptOutput",
    title: "Display",
    inputs: {
      value: () => new NodeInterface("Value", null)
    },
    outputs: {
      display: () => new TextInterface("Display", "null")
    },
    calculate(inputs) {
      let code;
      code = inputs.value ? inputs.value : "-1";
      return {
        display: code
      };
    }
  });  

const PromptInput = defineNode({
    type: "PromptInput",
    inputs: {
        source: () => new TextareaInputInterface("Text", "Hi!").setPort(false),
    },
    outputs: {
        output: () => new NodeInterface<string>("Output", "null"),
    },
    calculate(inputs, { globalValues }) {
        return {output:inputs.source};
    },
});

const CodeBlockExtract = defineNode({
    type: "CodeBlockExtract",
    inputs: {
        source: () => new TextInputInterface("Text", ""),
        header: () => new TextInputInterface("Text", "python").setPort(false),
    },
    outputs: {
        output: () => new NodeInterface<string>("Output", "null"),
    },
    calculate(inputs, { globalValues }) {
        return {output:inputs.header};
    },
});

export {PromptInput,PromptOutput,CodeBlockExtract};