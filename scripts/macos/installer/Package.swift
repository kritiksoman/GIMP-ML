// swift-tools-version:6.2
import PackageDescription

let package = Package(
    name: "GIMPMLInstaller",
    platforms: [.macOS(.v13)],
    products: [.executable(name: "GIMPMLInstaller", targets: ["GIMPMLInstaller"])],
    dependencies: [
        .package(url: "https://github.com/migueldeicaza/SwiftTerm.git", exact: "1.99.2")
    ],
    targets: [
        .executableTarget(
            name: "GIMPMLInstaller",
            dependencies: [.product(name: "SwiftTerm", package: "SwiftTerm")]
        )
    ],
    swiftLanguageModes: [.v5]
)
