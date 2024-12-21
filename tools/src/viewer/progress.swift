import AppKit
import Foundation

class SSEEvent {
    let timestamp: Date
    let message: String
    
    init(timestamp: Date = Date(), message: String) {
        self.timestamp = timestamp
        self.message = message
    }
}

class SSEClient {
    var events: [SSEEvent] = []
    var isConnected = false
    var statusMessage = "Not connected"
    var onEventReceived: ((SSEEvent) -> Void)?
    var onStatusChanged: ((String) -> Void)?
    
    private var urlSession: URLSession?
    private var eventSource: URLSessionDataTask?
    private var buffer = ""
    
    func connect(to urlString: String) {
        guard let url = URL(string: urlString) else {
            addEvent("Invalid URL")
            return
        }
        
        disconnect()
        
        let configuration = URLSessionConfiguration.default
        configuration.timeoutIntervalForRequest = TimeInterval(Float.infinity)
        urlSession = URLSession(configuration: configuration)
        
        var request = URLRequest(url: url)
        request.setValue("text/event-stream", forHTTPHeaderField: "Accept")
        
        eventSource = urlSession?.dataTask(with: request) { [weak self] data, response, error in
            if let error = error {
                DispatchQueue.main.async {
                    self?.addEvent("Error: \(error.localizedDescription)")
                    self?.disconnect()
                }
                return
            }
            
            if let data = data, let text = String(data: data, encoding: .utf8) {
                self?.processSSEData(text)
            }
        }
        
        isConnected = true
        updateStatus("Connected to \(urlString)")
        eventSource?.resume()
    }
    
    func disconnect() {
        eventSource?.cancel()
        eventSource = nil
        urlSession?.invalidateAndCancel()
        urlSession = nil
        
        isConnected = false
        updateStatus("Disconnected")
    }
    
    private func processSSEData(_ text: String) {
        buffer += text
        
        while let lineEnd = buffer.range(of: "\n") {
            let line = String(buffer[..<lineEnd.lowerBound])
            buffer.removeSubrange(..<lineEnd.upperBound)
            
            if line.hasPrefix("data:") {
                let data = String(line.dropFirst(5)).trimmingCharacters(in: .whitespaces)
                DispatchQueue.main.async {
                    self.addEvent(data)
                }
            }
        }
    }
    
    private func addEvent(_ message: String) {
        let event = SSEEvent(message: message)
        events.append(event)
        onEventReceived?(event)
    }
    
    private func updateStatus(_ message: String) {
        statusMessage = message
        onStatusChanged?(message)
    }
}

class MainWindowController: NSWindowController {
    let sseClient = SSEClient()
    let urlField = NSTextField()
    let connectButton = NSButton()
    let textView = NSTextView()
    let statusLabel = NSTextField()
    
    convenience init() {
        let window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 600, height: 400),
            styleMask: [.titled, .closable, .miniaturizable, .resizable],
            backing: .buffered,
            defer: false
        )
        window.title = "SSE Event Tracker"
        window.setFrameAutosaveName("Main Window")
        
        self.init(window: window)
        
        window.delegate = self
        setupUI()
        setupSSEClient()
    }
    
    private func setupUI() {
        guard let contentView = window?.contentView else { return }
        
        // URL input and connect button
        urlField.stringValue = "http://localhost:5000/events"
        urlField.frame = NSRect(x: 20, y: 350, width: 400, height: 24)
        urlField.placeholderString = "SSE URL"
        contentView.addSubview(urlField)
        
        connectButton.title = "Connect"
        connectButton.bezelStyle = .rounded
        connectButton.frame = NSRect(x: 430, y: 350, width: 100, height: 24)
        connectButton.target = self
        connectButton.action = #selector(toggleConnection)
        contentView.addSubview(connectButton)
        
        // Text view for events
        let scrollView = NSScrollView(frame: NSRect(x: 20, y: 40, width: 560, height: 280))
        scrollView.hasVerticalScroller = true
        scrollView.hasHorizontalScroller = false
        scrollView.autoresizingMask = [.width, .height]
        
        textView.frame = scrollView.bounds
        textView.autoresizingMask = [.width, .height]
        textView.isEditable = false
        textView.font = NSFont.monospacedSystemFont(ofSize: 12, weight: .regular)
        
        scrollView.documentView = textView
        contentView.addSubview(scrollView)
        
        // Status label
        statusLabel.stringValue = "Not connected"
        statusLabel.isEditable = false
        statusLabel.isBezeled = false
        statusLabel.drawsBackground = false
        statusLabel.frame = NSRect(x: 20, y: 10, width: 560, height: 20)
        contentView.addSubview(statusLabel)
    }
    
    private func setupSSEClient() {
        sseClient.onEventReceived = { [weak self] event in
            let timestamp = DateFormatter.localizedString(from: event.timestamp, dateStyle: .none, timeStyle: .medium)
            let text = "[\(timestamp)] \(event.message)\n"
            self?.textView.string += text
            self?.textView.scrollToEndOfDocument(nil)
        }
        
        sseClient.onStatusChanged = { [weak self] status in
            self?.statusLabel.stringValue = status
            self?.connectButton.title = self?.sseClient.isConnected == true ? "Disconnect" : "Connect"
        }
    }
    
    @objc private func toggleConnection() {
        if sseClient.isConnected {
            sseClient.disconnect()
        } else {
            sseClient.connect(to: urlField.stringValue)
        }
    }
}

extension MainWindowController: NSWindowDelegate {
    func windowWillClose(_ notification: Notification) {
        NSApplication.shared.terminate(self)
    }
}

class AppDelegate: NSObject, NSApplicationDelegate {
    var mainWindowController: MainWindowController?
    
    func applicationDidFinishLaunching(_ notification: Notification) {
        mainWindowController = MainWindowController()
        mainWindowController?.window?.center()
        mainWindowController?.showWindow(nil)
    }
    
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        return true
    }
}

// Main entry point
let app = NSApplication.shared
let delegate = AppDelegate()
app.delegate = delegate
app.run()