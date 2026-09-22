// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "LayaBrowser",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "LayaEngine", targets: ["LayaEngine"]),
        .executable(name: "LayaServer", targets: ["LayaServer"]),
    ],
    targets: [
        .target(name: "LayaEngine"),
        .executableTarget(name: "LayaServer", dependencies: ["LayaEngine"]),
        .testTarget(name: "LayaEngineTests", dependencies: ["LayaEngine"]),
    ]
)
