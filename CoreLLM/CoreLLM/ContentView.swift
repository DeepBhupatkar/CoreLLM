//
//  ContentView.swift
//  CoreLLM
//
//  Created by DEEP BHUPATKAR on 17/05/25.
//

import SwiftUI
import MLX
import MLXLLM
import MLXLMCommon
import MLXRandom
import MarkdownUI
import Metal
import Tokenizers

enum DisplayStyle: String, CaseIterable, Identifiable {
    case plain, markdown
    var id: Self { self }
}

struct ContentView: View {
    // LMMOdel
    @StateObject private var viewModel = LLMModelManager.LLMEvaluator()
    @State private var messageText: String = ""
    @State private var selectedDisplayStyle = DisplayStyle.markdown
    @State private var selectedModel: LLMModelOption = .smolLM_135M_4bit
    @Namespace private var bottomID  // For auto-scrolling

    var body: some View {
        VStack(spacing: 0) {
            // Status Bar
            HStack {
                Text(viewModel.modelInfo)
                    .font(.caption)
                    .foregroundColor(.secondary)
                
                if case .error(let error) = viewModel.loadState {
                    Button(action: {
                        Task {
                            viewModel.setModel(selectedModel)
                            try? await viewModel.load()
                        }
                    }) {
                        Image(systemName: "arrow.clockwise.circle")
                            .foregroundColor(.blue)
                    }
                }
                
                Spacer()
                
                // Model selector dropdown
                Picker("Model", selection: $selectedModel) {
                    ForEach(LLMModelOption.allCases) { model in
                        Text(model.displayName).tag(model)
                    }
                }
                .pickerStyle(.menu)
                .onChange(of: selectedModel) { newModel in
                    Task {
                        viewModel.setModel(newModel)
                        try? await viewModel.load()
                    }
                }
            }
            .padding(8)
            .background(Color.gray.opacity(0.1))
            
            // Chat Messages
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(spacing: 16) {
                        ForEach(viewModel.messages) { message in
                            MessageBubble(message: message, displayStyle: $selectedDisplayStyle)
                        }
                        Color.clear.frame(height: 1).id(bottomID) // Scroll anchor
                    }
                    .padding()
                }
                .onChange(of: viewModel.messages.count) { _ in
                    withAnimation {
                        proxy.scrollTo(bottomID, anchor: .bottom)
                    }
                }
            }

            // Input Area
            HStack(spacing: 12) {
                TextField("Type a message...", text: $messageText, onCommit: sendMessage)
                    .textFieldStyle(.roundedBorder)
                    .disabled(viewModel.running)

                Button(action: sendMessage) {
                    Image(systemName: "arrow.up.circle.fill")
                        .font(.system(size: 30))
                        .foregroundColor(viewModel.running ? .gray : .blue)
                }
                .disabled(viewModel.running || messageText.isEmpty)
            }
            .padding()
        }
        .padding()
        .toolbar {
            ToolbarItem(placement: .primaryAction) {
                Button(action: { copyToClipboard(viewModel.output) }) {
                    Label("Copy Output", systemImage: "doc.on.doc.fill")
                }
                .disabled(viewModel.output.isEmpty)
            }

            ToolbarItem(placement: .primaryAction) {
                Picker("Display Style", selection: $selectedDisplayStyle) {
                    ForEach(DisplayStyle.allCases) { style in
                        Text(style.rawValue.capitalized)
                    }
                }
                .pickerStyle(.segmented)
                .frame(width: 150)
            }
        }
        .task {
            self.messageText = viewModel.modelConfiguration.defaultPrompt
            try? await viewModel.load()
        }
    }

    private func sendMessage() {
        guard !messageText.isEmpty else { return }
        let text = messageText
        messageText = ""  // Clear input field instantly
        
        Task {
            await viewModel.generate(prompt: text)
        }
    }

    private func copyToClipboard(_ string: String) {
        #if os(macOS)
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(string, forType: .string)
        #else
        UIPasteboard.general.string = string
        #endif
    }
}

struct MessageBubble: View {
    let message: LLMModelManager.LLMEvaluator.Message
    @Binding var displayStyle: DisplayStyle

    var body: some View {
        HStack {
            if message.isUser {
                Spacer()
                Text(message.content)
                    .padding()
                    .background(Color.blue)
                    .foregroundColor(.white)
                    .cornerRadius(12)
            } else {
                HStack(alignment: .bottom, spacing: 8) {
                    Image(systemName: "sparkles")
                        .foregroundColor(.gray)
                    
                    Group {
                        if displayStyle == .markdown {
                            Markdown(message.content)
                                .textSelection(.enabled)
                        } else {
                            Text(message.content)
                        }
                    }
                    .padding()
                    .background(Color.gray.opacity(0.1))
                    .cornerRadius(12)
                    
                    Spacer()
                }
            }
        }
        .padding(.vertical, 4)
    }
}
