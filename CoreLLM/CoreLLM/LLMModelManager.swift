//
//  LLMModelManager.swift
//  CoreLLM
//
//  Created by DEEP BHUPATKAR on 17/05/25.
//

import Foundation
import MLX
import MLXLMCommon
import MLXLLM
import Tokenizers
import Hub
import MLXRandom
import Network

/// Manages the LLM loading and generation
class LLMModelManager {
    @MainActor
    class LLMEvaluator: ObservableObject {
        // Observable properties
        @Published var running = false
        @Published var includeWeatherTool = false
        @Published var output = ""
        @Published var modelInfo = ""
        @Published var stat = ""
        @Published var messages: [Message] = []
        @Published var loadState = LoadState.idle
        
        struct Message: Identifiable, Equatable {
            let id = UUID()
            let content: String
            let isUser: Bool
        }
        
        enum LoadState {
            case idle
            case loading
            case loaded(ModelContainer)
            case error(Error)
        }

        var modelConfiguration = LLMRegistry.smolLM_135M_4bit
        let generateParameters = GenerateParameters(
            temperature: 0.7,
            topP: 0.95,
            repetitionPenalty: 1.0,
            repetitionContextSize: 64
        )
        let maxTokens = 240
        let displayEveryNTokens = 4
        
        // Function to set the model from the dropdown selection
        func setModel(_ modelOption: LLMModelOption) {
            // Reset the load state and clear old model data
            loadState = .idle
            
            // Set the model configuration based on the selected option
            switch modelOption {
            case .codeLlama13b4bit:
                modelConfiguration = LLMRegistry.codeLlama13b4bit
            case .deepSeekR1_7B_4bit:
                modelConfiguration = LLMRegistry.deepSeekR1_7B_4bit
            case .gemma2bQuantized:
                modelConfiguration = LLMRegistry.gemma2bQuantized
            case .gemma_2_2b_it_4bit:
                modelConfiguration = LLMRegistry.gemma_2_2b_it_4bit
            case .gemma_2_9b_it_4bit:
                modelConfiguration = LLMRegistry.gemma_2_9b_it_4bit
            case .llama3_1_8B_4bit:
                modelConfiguration = LLMRegistry.llama3_1_8B_4bit
            case .llama3_2_1B_4bit:
                modelConfiguration = LLMRegistry.llama3_2_1B_4bit
            case .llama3_2_3B_4bit:
                modelConfiguration = LLMRegistry.llama3_2_3B_4bit
            case .llama3_8B_4bit:
                modelConfiguration = LLMRegistry.llama3_8B_4bit
            case .mistral7B4bit:
                modelConfiguration = LLMRegistry.mistral7B4bit
            case .mistralNeMo4bit:
                modelConfiguration = LLMRegistry.mistralNeMo4bit
            case .openelm270m4bit:
                modelConfiguration = LLMRegistry.openelm270m4bit
            case .phi3_5MoE:
                modelConfiguration = LLMRegistry.phi3_5MoE
            case .phi3_5_4bit:
                modelConfiguration = LLMRegistry.phi3_5_4bit
            case .phi4bit:
                modelConfiguration = LLMRegistry.phi4bit
            case .qwen205b4bit:
                modelConfiguration = LLMRegistry.qwen205b4bit
            case .qwen2_5_7b:
                modelConfiguration = LLMRegistry.qwen2_5_7b
            case .qwen2_5_1_5b:
                modelConfiguration = LLMRegistry.qwen2_5_1_5b
            case .smolLM_135M_4bit:
                modelConfiguration = LLMRegistry.smolLM_135M_4bit
            case .gemma3_1B_4bit:
                modelConfiguration = LLMRegistry.gemma3_1B_4bit
            }
        }

        /// load and return the model -- can be called multiple times, subsequent calls will
        /// just return the loaded model
        func load() async throws -> ModelContainer {
            loadState = .loading
            // limit the buffer cache
            MLX.GPU.set(cacheLimit: 20 * 1024 * 1024)

            let modelContainer = try await LLMModelFactory.shared.loadContainer(
                configuration: modelConfiguration
            ) {
                [modelConfiguration] progress in
                Task { @MainActor in
                    self.modelInfo =
                        "Downloading \(modelConfiguration.name): \(Int(progress.fractionCompleted * 100))%"
                }
            }
            let numParams = await modelContainer.perform { context in
                context.model.numParameters()
            }

            self.modelInfo =
                "Loaded \(modelConfiguration.id).  Weights: \(numParams / (1024*1024))M"
            loadState = .loaded(modelContainer)
            return modelContainer
        }

        func generate(prompt: String) async {
            messages.append(Message(content: prompt, isUser: true))
            let thinkingMessage = Message(content: "Thinking...", isUser: false)
            messages.append(thinkingMessage)

            guard !running else { return }
            running = true

            do {
                // Check if the model is already loaded
                let modelContainer: ModelContainer
                if case .loaded(let loadedModel) = loadState {
                    modelContainer = loadedModel
                } else {
                    modelContainer = try await load()
                }

                MLXRandom.seed(UInt64(Date.timeIntervalSinceReferenceDate * 1000))

                let result = try await modelContainer.perform { context in
                    let input = try await context.processor.prepare(
                        input: .init(
                            messages: [
                                ["role": "system", "content": "You are a helpful assistant."],
                                ["role": "user", "content": prompt],
                            ]))
                    return try MLXLMCommon.generate(
                        input: input, parameters: generateParameters, context: context
                    ) { tokens in
                        if tokens.count % displayEveryNTokens == 0 {
                            let text = context.tokenizer.decode(tokens: tokens)
                            Task { @MainActor [weak self] in
                                guard let self = self else { return }
                                self.output = text
                                
                                let lastIndex = self.messages.count - 1
                                if lastIndex >= 0 && !self.messages[lastIndex].isUser {
                                    self.messages[lastIndex] = Message(
                                        content: text.isEmpty ? "..." : text,
                                        isUser: false
                                    )
                                }
                            }
                        }
                        if tokens.count >= maxTokens {
                            return .stop
                        } else {
                            return .more
                        }
                    }
                }

                if result.output != self.output {
                    self.output = result.output
                    let lastIndex = self.messages.count - 1
                    if lastIndex >= 0 && !self.messages[lastIndex].isUser {
                        self.messages[lastIndex] = Message(content: result.output, isUser: false)
                    }
                }
                self.stat = " Tokens/second: \(String(format: "%.3f", result.tokensPerSecond))"

            } catch {
                output = "Failed: \(error)"
                loadState = .error(error)
                let lastIndex = self.messages.count - 1
                if lastIndex >= 0 && !self.messages[lastIndex].isUser {
                    self.messages[lastIndex] = Message(
                        content: "Error: \(error.localizedDescription)",
                        isUser: false
                    )
                }
            }

            running = false
        }

    }
}
    

