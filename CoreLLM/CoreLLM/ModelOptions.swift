//
//  ModelOptions.swift
//  CoreLLM
//
//  Created by DEEP BHUPATKAR on 21/05/25.
//

import Foundation

// Enum for model options
enum LLMModelOption: String, CaseIterable, Identifiable {
    case codeLlama13b4bit
    case deepSeekR1_7B_4bit
    case gemma2bQuantized
    case gemma_2_2b_it_4bit
    case gemma_2_9b_it_4bit
    case llama3_1_8B_4bit
    case llama3_2_1B_4bit
    case llama3_2_3B_4bit
    case llama3_8B_4bit
    case mistral7B4bit
    case mistralNeMo4bit
    case openelm270m4bit
    case phi3_5MoE
    case phi3_5_4bit
    case phi4bit
    case qwen205b4bit
    case qwen2_5_7b
    case qwen2_5_1_5b
    case smolLM_135M_4bit
    case gemma3_1B_4bit
    
    var id: Self { self }
    
    var displayName: String {
        switch self {
        case .codeLlama13b4bit: return "CodeLlama 13B (4-bit)"
        case .deepSeekR1_7B_4bit: return "DeepSeek R1 7B (4-bit)"
        case .gemma2bQuantized: return "Gemma 2B Quantized"
        case .gemma_2_2b_it_4bit: return "Gemma 2 2B IT (4-bit)"
        case .gemma_2_9b_it_4bit: return "Gemma 2 9B IT (4-bit)"
        case .llama3_1_8B_4bit: return "Llama 3 1.8B (4-bit)"
        case .llama3_2_1B_4bit: return "Llama 3 2.1B (4-bit)"
        case .llama3_2_3B_4bit: return "Llama 3 2.3B (4-bit)"
        case .llama3_8B_4bit: return "Llama 3 8B (4-bit)"
        case .mistral7B4bit: return "Mistral 7B (4-bit)"
        case .mistralNeMo4bit: return "Mistral NeMo (4-bit)"
        case .openelm270m4bit: return "OpenELM 270M (4-bit)"
        case .phi3_5MoE: return "Phi 3.5 MoE"
        case .phi3_5_4bit: return "Phi 3.5 (4-bit)"
        case .phi4bit: return "Phi (4-bit)"
        case .qwen205b4bit: return "Qwen 2 0.5B (4-bit)"
        case .qwen2_5_7b: return "Qwen 2 5.7B"
        case .qwen2_5_1_5b: return "Qwen 2 5.1 5B"
        case .smolLM_135M_4bit: return "SmolLM 135M (4-bit)"
        case .gemma3_1B_4bit: return "Gemma 3 1B (4-bit)"
        }
    }
    
    var registryValue: String {
        return self.rawValue
    }
}
