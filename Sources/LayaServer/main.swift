@preconcurrency import Network
import Foundation
import LayaEngine

private let maxRequestBytes = 512 * 1024

private struct HTTPRequest {
    let method: String
    let path: String
    let headers: [String: String]
    let body: Data

    static func parse(_ data: Data) throws -> HTTPRequest? {
        let delimiter = Data("\r\n\r\n".utf8)
        guard let headerEnd = data.range(of: delimiter) else { return nil }

        let headerText = String(decoding: data[..<headerEnd.lowerBound], as: UTF8.self)
        let lines = headerText.components(separatedBy: "\r\n")
        guard let requestLine = lines.first else {
            throw HTTPError.badRequest("Missing request line")
        }
        let requestParts = requestLine.split(separator: " ", maxSplits: 2).map(String.init)
        guard requestParts.count == 3 else {
            throw HTTPError.badRequest("Malformed request line")
        }

        var headers: [String: String] = [:]
        for line in lines.dropFirst() where !line.isEmpty {
            let parts = line.split(separator: ":", maxSplits: 1).map(String.init)
            guard parts.count == 2 else { throw HTTPError.badRequest("Malformed header") }
            headers[parts[0].trimmingCharacters(in: .whitespacesAndNewlines).lowercased()] =
                parts[1].trimmingCharacters(in: .whitespacesAndNewlines)
        }

        let contentLength = Int(headers["content-length"] ?? "0") ?? 0
        guard contentLength >= 0, contentLength <= maxRequestBytes else {
            throw HTTPError.payloadTooLarge
        }
        let bodyStart = headerEnd.upperBound
        guard data.count >= bodyStart + contentLength else { return nil }

        let rawPath = requestParts[1]
        let path = rawPath.split(separator: "?", maxSplits: 1).first.map(String.init) ?? rawPath
        return HTTPRequest(
            method: requestParts[0].uppercased(), path: path, headers: headers,
            body: Data(data[bodyStart..<(bodyStart + contentLength)]))
    }
}

private enum HTTPError: Error {
    case badRequest(String)
    case payloadTooLarge
}

private struct ErrorPayload: Encodable {
    let error: String
}

private struct TetrisScoreRequest: Decodable {
    let states: [String]
}

private struct TetrisScore: Encodable {
    let pTrue: Float
    let pFalse: Float
    let probabilities: [Float]
    let tokenCount: Int
    let bucketLength: Int
    let stateWasTruncated: Bool
    let latencyMs: Double
}

private struct TetrisScoreResponse: Encodable {
    let model: String
    let bucket: Int
    let results: [TetrisScore]
    let elapsedMs: Double
}

private struct HealthResponse: Encodable {
    let status: String
    let model: String
    let bucket: Int
}

private struct HTTPResponse {
    let status: Int
    let reason: String
    let contentType: String
    let body: Data

    static func json<T: Encodable>(_ value: T, status: Int = 200, reason: String? = nil) -> HTTPResponse {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let body = (try? encoder.encode(value)) ?? Data("{\"error\":\"encoding failed\"}".utf8)
        return HTTPResponse(
            status: status,
            reason: reason ?? HTTPResponse.reason(for: status),
            contentType: "application/json; charset=utf-8",
            body: body)
    }

    static func text(_ value: String, status: Int = 200, reason: String? = nil) -> HTTPResponse {
        HTTPResponse(
            status: status,
            reason: reason ?? HTTPResponse.reason(for: status),
            contentType: "text/plain; charset=utf-8",
            body: Data(value.utf8))
    }

    private static func reason(for status: Int) -> String {
        switch status {
        case 200: return "OK"
        case 400: return "Bad Request"
        case 404: return "Not Found"
        case 405: return "Method Not Allowed"
        case 413: return "Payload Too Large"
        case 500: return "Internal Server Error"
        default: return "Error"
        }
    }
}

private final class LayaHTTPServer: @unchecked Sendable {
    private let listener: NWListener
    private let manager: LayaManager
    private let queue = DispatchQueue(label: "laya.http", qos: .userInitiated)

    init(manager: LayaManager, port: UInt16) throws {
        self.manager = manager
        let parameters = NWParameters.tcp
        parameters.requiredInterfaceType = .loopback
        self.listener = try NWListener(using: parameters, on: NWEndpoint.Port(rawValue: port)!)
    }

    func start() {
        listener.stateUpdateHandler = { state in
            switch state {
            case .ready:
                print("Laya API listening at http://127.0.0.1:\(Self.portDescription(self.listener))")
            case .failed(let error):
                fputs("Laya API failed: \(error.localizedDescription)\n", stderr)
                Foundation.exit(1)
            default:
                break
            }
        }
        listener.newConnectionHandler = { [weak self] connection in
            self?.accept(connection)
        }
        listener.start(queue: queue)
    }

    private static func portDescription(_ listener: NWListener) -> String {
        listener.port.map { String(describing: $0) } ?? "unknown"
    }

    private func accept(_ connection: NWConnection) {
        connection.stateUpdateHandler = { state in
            if case .failed = state { connection.cancel() }
        }
        connection.start(queue: queue)
        receive(connection, buffer: Data())
    }

    private func receive(_ connection: NWConnection, buffer: Data) {
        connection.receive(minimumIncompleteLength: 1, maximumLength: maxRequestBytes) {
            [weak self] data, _, isComplete, error in
            guard let self else {
                connection.cancel()
                return
            }
            var buffer = buffer
            if let data { buffer.append(data) }
            if buffer.count > maxRequestBytes {
                self.send(.json(ErrorPayload(error: "request too large"), status: 413), over: connection)
                return
            }
            do {
                if let request = try HTTPRequest.parse(buffer) {
                    Task { [weak self] in
                        guard let self else { return }
                        let response = await self.handle(request)
                        self.send(response, over: connection)
                    }
                } else if isComplete || error != nil {
                    self.send(.json(ErrorPayload(error: "incomplete HTTP request"), status: 400), over: connection)
                } else {
                    self.receive(connection, buffer: buffer)
                }
            } catch HTTPError.payloadTooLarge {
                self.send(.json(ErrorPayload(error: "request too large"), status: 413), over: connection)
            } catch {
                self.send(.json(ErrorPayload(error: error.localizedDescription), status: 400), over: connection)
            }
        }
    }

    private func send(_ response: HTTPResponse, over connection: NWConnection) {
        let header =
            "HTTP/1.1 \(response.status) \(response.reason)\r\n"
            + "Content-Type: \(response.contentType)\r\n"
            + "Content-Length: \(response.body.count)\r\n"
            + "Connection: close\r\n"
            + "Cache-Control: no-store\r\n"
            + "\r\n"
        var payload = Data(header.utf8)
        payload.append(response.body)
        connection.send(content: payload, completion: .contentProcessed { _ in connection.cancel() })
    }

    private func handle(_ request: HTTPRequest) async -> HTTPResponse {
        if request.method == "GET", request.path == "/healthz" {
            return .json(HealthResponse(status: "ok", model: "laya-multilingual", bucket: manager.lengths.first ?? 0))
        }
        guard request.method == "POST" else {
            return .json(ErrorPayload(error: "method not allowed"), status: 405)
        }
        guard request.path == "/v1/laya/tetris/score" else {
            return .json(ErrorPayload(error: "not found"), status: 404)
        }
        guard request.headers["content-type"]?.lowercased().contains("application/json") == true else {
            return .json(ErrorPayload(error: "Content-Type must be application/json"), status: 400)
        }

        do {
            let payload = try JSONDecoder().decode(TetrisScoreRequest.self, from: request.body)
            guard !payload.states.isEmpty, payload.states.count <= 64 else {
                return .json(ErrorPayload(error: "states must contain 1 to 64 items"), status: 400)
            }
            guard payload.states.allSatisfy({ !$0.isEmpty && $0.count <= 4_000 }) else {
                return .json(ErrorPayload(error: "each state must contain 1 to 4,000 characters"), status: 400)
            }
            let question = LayaQuestion.noul("Is this a clean placement?")
            let started = DispatchTime.now().uptimeNanoseconds
            var results: [TetrisScore] = []
            results.reserveCapacity(payload.states.count)
            for state in payload.states {
                let callStarted = DispatchTime.now().uptimeNanoseconds
                let answer = try await manager.answer(state: state, question: question)
                let latency = Double(DispatchTime.now().uptimeNanoseconds - callStarted) / 1_000_000
                let pTrue = answer.noul ?? answer.probabilities.last ?? 0
                let pFalse = answer.probabilities.first ?? (1 - pTrue)
                results.append(
                    TetrisScore(
                        pTrue: pTrue, pFalse: pFalse, probabilities: answer.probabilities,
                        tokenCount: answer.tokenCount, bucketLength: answer.bucketLength,
                        stateWasTruncated: answer.stateWasTruncated, latencyMs: latency))
            }
            let elapsed = Double(DispatchTime.now().uptimeNanoseconds - started) / 1_000_000
            return .json(
                TetrisScoreResponse(
                    model: "laya-multilingual", bucket: manager.lengths.first ?? 0, results: results,
                    elapsedMs: elapsed))
        } catch {
            return .json(ErrorPayload(error: error.localizedDescription), status: 400)
        }
    }
}

private final class ServerLifetime: @unchecked Sendable {
    var server: LayaHTTPServer?
}

@main
struct LayaServerMain {
    static func main() {
        let lifetime = ServerLifetime()
        Task { await run(lifetime: lifetime) }
        withExtendedLifetime(lifetime) { dispatchMain() }
    }

    private static func run(lifetime: ServerLifetime) async {
        do {
            let environment = ProcessInfo.processInfo.environment
            let precision = environment["LAYA_PRECISION"] ?? "fp16"
            let configuration = LayaManager.Configuration(lengths: [128], precision: precision)
            let manager: LayaManager
            if let path = environment["LAYA_MODEL_DIR"], !path.isEmpty {
                manager = try await LayaManager.load(
                    from: URL(fileURLWithPath: path, isDirectory: true), configuration: configuration)
            } else {
                let directory = try await LayaModelStore.ensure(lengths: [128], precision: precision)
                manager = try await LayaManager.load(from: directory, configuration: configuration)
            }
            _ = try await manager.answer(
                state: "The piece leaves no holes and keeps the stack low.",
                question: .noul("Is this a clean placement?"))

            let port = UInt16(environment["LAYA_PORT"].flatMap(UInt16.init) ?? 8787)
            let server = try LayaHTTPServer(manager: manager, port: port)
            print("Loaded laya-multilingual L128 (\(precision))")
            lifetime.server = server
            server.start()
        } catch {
            fputs("Unable to start Laya API: \(error.localizedDescription)\n", stderr)
            Foundation.exit(1)
        }
    }
}
