import AppKit
import SwiftTerm
import UniformTypeIdentifiers

@MainActor
final class Installer: NSObject, NSApplicationDelegate, NSWindowDelegate,
                       LocalProcessTerminalViewDelegate {
    private var window: NSWindow!
    private let terminal = LocalProcessTerminalView(frame: .zero)
    private let progress = NSProgressIndicator()
    private let status = NSTextField(wrappingLabelWithString:
        "Fully quit GIMP (Cmd+Q). Requires Python 3.10+ and internet access. No administrator password is needed.")
    private let location = NSTextField(labelWithString: "/Applications/GIMP.app")
    private let installButton = NSButton(title: "Install", target: nil, action: nil)
    private let chooseButton = NSButton(title: "Choose GIMP…", target: nil, action: nil)
    private var gimp = URL(fileURLWithPath: "/Applications/GIMP.app")
    private var session: URL?
    private var timer: Timer?
    private var installing = false
    private var started = false

    func applicationDidFinishLaunching(_ notification: Notification) {
        let menu = NSMenu()
        let appItem = NSMenuItem()
        menu.addItem(appItem)
        let appMenu = NSMenu()
        appMenu.addItem(withTitle: "Quit Installer", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")
        appItem.submenu = appMenu
        let editItem = NSMenuItem()
        menu.addItem(editItem)
        let editMenu = NSMenu(title: "Edit")
        editMenu.addItem(withTitle: "Copy", action: #selector(NSText.copy(_:)), keyEquivalent: "c")
        editMenu.addItem(withTitle: "Paste", action: #selector(NSText.paste(_:)), keyEquivalent: "v")
        editItem.submenu = editMenu
        NSApp.mainMenu = menu

        window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 820, height: 560),
                          styleMask: [.titled, .closable, .miniaturizable, .resizable],
                          backing: .buffered, defer: false)
        window.title = "Install GIMP-ML"
        window.minSize = NSSize(width: 650, height: 430)
        window.isReleasedWhenClosed = false
        window.delegate = self
        guard let content = window.contentView else { return }
        let stack = NSStackView()
        stack.orientation = .vertical
        stack.alignment = .leading
        stack.spacing = 12
        stack.translatesAutoresizingMaskIntoConstraints = false
        content.addSubview(stack)
        NSLayoutConstraint.activate([
            stack.leadingAnchor.constraint(equalTo: content.leadingAnchor, constant: 20),
            stack.trailingAnchor.constraint(equalTo: content.trailingAnchor, constant: -20),
            stack.topAnchor.constraint(equalTo: content.topAnchor, constant: 20),
            stack.bottomAnchor.constraint(equalTo: content.bottomAnchor, constant: -20)
        ])
        let title = NSTextField(labelWithString: "GIMP-ML Setup")
        title.font = .boldSystemFont(ofSize: 20)
        stack.addArrangedSubview(title)
        stack.addArrangedSubview(status)
        status.widthAnchor.constraint(equalTo: stack.widthAnchor).isActive = true
        chooseButton.target = self
        chooseButton.action = #selector(chooseGIMP)
        location.lineBreakMode = .byTruncatingMiddle
        let row = NSStackView(views: [location, chooseButton])
        row.orientation = .horizontal
        stack.addArrangedSubview(row)
        row.widthAnchor.constraint(equalTo: stack.widthAnchor).isActive = true
        progress.style = .bar
        progress.isIndeterminate = true
        progress.isDisplayedWhenStopped = false
        stack.addArrangedSubview(progress)
        progress.widthAnchor.constraint(equalTo: stack.widthAnchor).isActive = true
        let terminalContainer = NSView()
        stack.addArrangedSubview(terminalContainer)
        terminalContainer.widthAnchor.constraint(equalTo: stack.widthAnchor).isActive = true
        terminalContainer.heightAnchor.constraint(greaterThanOrEqualToConstant: 220).isActive = true
        terminal.frame = terminalContainer.bounds
        terminal.autoresizingMask = [.width, .height]
        terminal.processDelegate = self
        terminal.font = .monospacedSystemFont(ofSize: 13, weight: .regular)
        terminalContainer.addSubview(terminal)
        terminal.feed(text: "Click Install to start setup.\r\nOutput and an interactive shell will appear here. No Terminal window is opened.\r\n")
        installButton.target = self
        installButton.action = #selector(installOrClose)
        installButton.bezelStyle = .rounded
        stack.addArrangedSubview(installButton)
        window.center()
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }

    @objc private func chooseGIMP() {
        let panel = NSOpenPanel()
        panel.title = "Select your GIMP 3 application"
        panel.allowedContentTypes = [.applicationBundle]
        panel.allowsMultipleSelection = false
        panel.beginSheetModal(for: window) { [weak self] response in
            guard response == .OK, let url = panel.url, let self else { return }
            self.gimp = url
            self.location.stringValue = url.path
        }
    }

    @objc private func installOrClose() {
        if started {
            window.performClose(nil)
            return
        }
        guard let resources = Bundle.main.resourceURL else {
            showError("The installer resources are missing.")
            return
        }
        let runner = resources.appendingPathComponent("install-session.command")
        guard FileManager.default.fileExists(atPath: runner.path) else {
            showError("The installer is incomplete: install-session.command is missing.")
            return
        }
        do {
            let directory = FileManager.default.temporaryDirectory
                .appendingPathComponent("gimpml-install-\(UUID().uuidString)", isDirectory: true)
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false,
                                                     attributes: [.posixPermissions: 0o700])
            session = directory
            let library = try FileManager.default.url(for: .libraryDirectory, in: .userDomainMask,
                                                       appropriateFor: nil, create: false)
            let log = library.appendingPathComponent("Logs/GIMP-ML/install.log")
            let result = directory.appendingPathComponent("status")
            let command = ["/bin/bash", runner.path, resources.path, gimp.path, log.path, result.path]
                .map(shellQuote).joined(separator: " ")
                + "; if [ -f \(shellQuote(result.path)) ]; then exec /bin/bash --noprofile --norc -i; else exit 1; fi"
            started = true
            installing = true
            chooseButton.isEnabled = false
            installButton.isEnabled = false
            status.stringValue = "Installing… Python setup and dependency downloads may take a few minutes. Ctrl+C interrupts setup."
            progress.startAnimation(nil)
            timer = Timer.scheduledTimer(timeInterval: 0.2, target: self,
                                        selector: #selector(checkResult), userInfo: nil, repeats: true)
            terminal.startProcess(executable: "/bin/bash",
                                  args: ["--noprofile", "--norc", "-c", command])
            window.makeFirstResponder(terminal)
        } catch {
            finish(error: error.localizedDescription)
        }
    }

    private func shellQuote(_ value: String) -> String {
        "'" + value.replacingOccurrences(of: "'", with: "'\\''") + "'"
    }

    @objc private func checkResult() {
        guard installing, let session else { return }
        let file = session.appendingPathComponent("status")
        do {
            let value = try String(contentsOf: file, encoding: .utf8)
                .trimmingCharacters(in: .whitespacesAndNewlines)
            guard let code = Int(value), (0...255).contains(code) else {
                finish(error: "Setup returned an invalid exit status.")
                return
            }
            finish(error: code == 0 ? nil : "Setup exited with status \(code). See the terminal output above and ~/Library/Logs/GIMP-ML/install.log.")
        } catch let error as CocoaError where error.code == .fileReadNoSuchFile {
            // The runner publishes its exit status atomically when setup ends.
        } catch {
            finish(error: "Cannot read the installation result: \(error.localizedDescription)")
        }
    }

    private func finish(error: String?) {
        installing = false
        timer?.invalidate()
        timer = nil
        progress.stopAnimation(nil)
        installButton.title = started ? "Done" : "Install"
        installButton.isEnabled = true
        if let error {
            showError(error)
        } else {
            status.stringValue = "Installed. Open GIMP → Filters → GIMP-ML → Settings to save your API key. The terminal is now an interactive shell; click Done before ejecting the disk image."
        }
    }

    private func showError(_ message: String) {
        status.stringValue = "Installation failed: \(message)"
        let alert = NSAlert()
        alert.alertStyle = .critical
        alert.messageText = "Installation failed"
        alert.informativeText = message
        alert.beginSheetModal(for: window)
    }

    private func canClose() -> Bool {
        guard installing else { return true }
        let alert = NSAlert()
        alert.messageText = "Installation is still running"
        alert.informativeText = "Wait for setup to finish, or press Ctrl+C in the embedded terminal to interrupt it before closing."
        alert.beginSheetModal(for: window)
        return false
    }

    func windowShouldClose(_ sender: NSWindow) -> Bool { canClose() }
    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        canClose() ? .terminateNow : .terminateCancel
    }
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }

    func applicationWillTerminate(_ notification: Notification) {
        timer?.invalidate()
        terminal.process.terminate()
        if let session {
            do {
                try FileManager.default.removeItem(at: session)
            } catch {
                NSLog("Cannot remove installer session %@: %@", session.path, error.localizedDescription)
            }
        }
    }

    func sizeChanged(source: LocalProcessTerminalView, newCols: Int, newRows: Int) {}
    func setTerminalTitle(source: LocalProcessTerminalView, title: String) {}
    func hostCurrentDirectoryUpdate(source: TerminalView, directory: String?) {}
    func processTerminated(source: TerminalView, exitCode: Int32?) {
        checkResult()
        if installing {
            finish(error: "The terminal process ended before setup reported a result.")
        }
    }
    func processFailedToStart(source: TerminalView, error: LocalProcessError) {
        finish(error: "Cannot start the embedded terminal: \(error)")
    }
}

@main
struct Main {
    @MainActor
    static func main() {
        let app = NSApplication.shared
        let installer = Installer()
        app.delegate = installer
        app.setActivationPolicy(.regular)
        app.run()
    }
}
