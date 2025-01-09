import Foundation
import NaturalLanguage

class Document {
    let id: String
    let content: String
    var embedding: [Double]?

    init(id: String, content: String) {
        self.id = id
        self.content = content
    }
}

class RAGSystem {
    private var documents: [Document] = []
    private let embeddingModel: NLEmbedding

    init() {
        guard let model = NLEmbedding.wordEmbedding(for: .english) else {
            fatalError("Unable to load embedding model")
        }
        embeddingModel = model
    }

    func addDocument(_ document: Document) {
        let words = document.content.components(separatedBy: .whitespacesAndNewlines)
        let embeddings = words.compactMap { embeddingModel.vector(for: $0) }
        let averageEmbedding = average(embeddings)
        document.embedding = averageEmbedding
        documents.append(document)
    }

    func searchRelevantDocuments(for query: String, limit: Int = 3) -> [Document] {
        let queryEmbedding = getEmbedding(for: query)
        let sortedDocuments = documents.sorted { doc1, doc2 in
            guard let emb1 = doc1.embedding, let emb2 = doc2.embedding else { return false }
            return cosineSimilarity(queryEmbedding, emb1) > cosineSimilarity(queryEmbedding, emb2)
        }
        return Array(sortedDocuments.prefix(limit))
    }

    func generateResponse(for query: String) -> String {
        let relevantDocs = searchRelevantDocuments(for: query)
        let context = relevantDocs.map { $0.content }.joined(separator: " ")
        let prompt = """
        Context: \(context)

        Human: \(query)

        Assistant: Based on the given context, I will provide a concise and accurate answer to the question.
        """

        return callOllama(with: prompt)
    }

    private func callOllama(with prompt: String) -> String {
        let ollamaURL = URL(string: "http://localhost:11434/api/generate")!
        var request = URLRequest(url: ollamaURL)
        request.httpMethod = "POST"
        request.addValue("application/json", forHTTPHeaderField: "Content-Type")

        let parameters: [String: Any] = [
            "model": "llama3.2:latest",
            "prompt": prompt,
            "stream": false,
        ]

        request.httpBody = try? JSONSerialization.data(withJSONObject: parameters)

        let semaphore = DispatchSemaphore(value: 0)
        var responseText = ""

        let task = URLSession.shared.dataTask(with: request) { data, _, error in
            defer { semaphore.signal() }

            if let error = error {
                print("Error: \(error.localizedDescription)")
                return
            }

            guard let data = data else {
                print("No data received")
                return
            }

            if let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
               let response = json["response"] as? String {
                responseText = response
            } else {
                print("Failed to parse response")
            }
        }

        task.resume()
        semaphore.wait()

        return responseText
    }

    private func getEmbedding(for text: String) -> [Double] {
        let words = text.components(separatedBy: .whitespacesAndNewlines)
        let embeddings = words.compactMap { embeddingModel.vector(for: $0) }
        return average(embeddings)
    }

    private func average(_ vectors: [[Double]]) -> [Double] {
        guard !vectors.isEmpty else { return [] }
        let sum = vectors.reduce(into: Array(repeating: 0.0, count: vectors[0].count)) { result, vector in
            for (index, value) in vector.enumerated() {
                result[index] += value
            }
        }
        return sum.map { $0 / Double(vectors.count) }
    }

    private func cosineSimilarity(_ v1: [Double], _ v2: [Double]) -> Double {
        guard v1.count == v2.count else { return 0 }
        let dotProduct = zip(v1, v2).map(*).reduce(0, +)
        let magnitude1 = sqrt(v1.map { $0 * $0 }.reduce(0, +))
        let magnitude2 = sqrt(v2.map { $0 * $0 }.reduce(0, +))
        return dotProduct / (magnitude1 * magnitude2)
    }
}

func runCommand(_ command: String, workingDirectory: String? = nil) -> (String, String?) {
    let process = Process()
    process.launchPath = "/bin/zsh" // 使用 zsh 作为 shell
    process.arguments = ["-c", command] // 使用 -c 选项执行命令

    if let directory = workingDirectory {
        process.currentDirectoryPath = directory // 设置工作目录
    }

    let pipe = Pipe()
    process.standardOutput = pipe
    process.standardError = pipe

    process.launch()

    let data = pipe.fileHandleForReading.readDataToEndOfFile()
    process.waitUntilExit()

    if let output = String(data: data, encoding: .utf8) {
        return (command, output)
    }

    return (command, nil)
}

func ragCommand(_ command: String, workingDirectory: String? = nil) {
    let content = runCommand(command, workingDirectory: workingDirectory)

    if let output = content.1 {
        ragSystem.addDocument(Document(id: command, content: "\(content.0), \(output)"))
    }
}

func ragFile(path: String, ragSystem: RAGSystem) {
    do {
        // 读取文件内容
        let content = try String(contentsOfFile: path, encoding: .utf8)
        ragSystem.addDocument(Document(id: path, content: content))
    } catch {
        print("无法读取 README.md 文件: \(error)")
    }
}

let INITIAL_PROMPT = """
    You are a git commit message generator.
    Your task is to help the user write a good commit message.

    You will receive a summary of git log as first message from the user,
    a summary of git diff as the second message from the user
    and an optional hint for the commit message as the third message of the user.

    Take the whole conversation in consideration and suggest a good commit message.
    Never say anything that is not your proposed commit message, never appologize.

    - Use imperative
    - One line only
    - Be clear and concise
    - Follow standard commit message conventions
    - Do not put message in quotes
    - Put the most important changes first
    - Focus on the intent of the change, not just the code change. WHY, not how.
    - Avoid using "refactor" or "update" as they are too vague

    Always provide only the commit message as answer.
"""

let workingDirectory = "/Users/ronnie/Github/syncnext"
//let workingDirectory = "/Users/ronnie/Github/KSPlayer"

// Example usage
let ragSystem = RAGSystem()

if runCommand("git diff", workingDirectory: workingDirectory).1 != "" {
    ragCommand("git diff", workingDirectory: workingDirectory)
    ragCommand("git status", workingDirectory: workingDirectory)

    // Generating a response
    let query = INITIAL_PROMPT
    let response = ragSystem.generateResponse(for: query)

    print("Commit: \(response)")
} else {
    print("git diff is empty")
}
