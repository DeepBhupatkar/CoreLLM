//
//  CoreLLMApp.swift
//  CoreLLM
//
//  Created by DEEP BHUPATKAR on 17/05/25.
//

import SwiftUI
import MLX
import MLXLMCommon

@main
struct CoreLLMApp: App {
    // GPU cache clearing on app termination
    @Environment(\.scenePhase) var scenePhase
    
    var body: some Scene {
        WindowGroup {
            ContentView()
                .onChange(of: scenePhase) { _, newPhase in
                    if newPhase == .background {
                        // Clear GPU cache when app goes to background
                        MLX.GPU.clearCache()
                    }
                }
        }
    }
}
