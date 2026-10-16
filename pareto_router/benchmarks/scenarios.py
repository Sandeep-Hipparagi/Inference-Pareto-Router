"""100 diverse prompt scenarios across the 2D (Logic x Knowledge) space."""

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkScenario:
    id: int
    category: str
    prompt: str
    expected_tier_min: str
    expected_effort: str
    is_overthinking_candidate: bool


# 100 Scenarios spanning all 6 archetypes
SCENARIOS: list[BenchmarkScenario] = [
    # --- Category 1: Low Logic, Low Knowledge (Greetings, Chit-chat, Aesthetic Edits) [20 prompts] ---
    BenchmarkScenario(1, "Greeting", "Hello!", "compact", "none", True),
    BenchmarkScenario(2, "Greeting", "Good morning, how are you today?", "compact", "none", True),
    BenchmarkScenario(3, "Greeting", "Hi, what is your name?", "compact", "none", True),
    BenchmarkScenario(4, "Greeting", "Hey there! Hope you are having a nice day.", "compact", "none", True),
    BenchmarkScenario(5, "Greeting", "Thanks for your help earlier!", "compact", "none", True),
    BenchmarkScenario(6, "Greeting", "Thank you so much! That worked.", "compact", "none", True),
    BenchmarkScenario(7, "Greeting", "Hi bot, who created you?", "compact", "none", True),
    BenchmarkScenario(8, "Greeting", "Good evening!", "compact", "none", True),
    BenchmarkScenario(9, "Greeting", "Howdy partner!", "compact", "none", True),
    BenchmarkScenario(10, "Greeting", "What can you help me with today?", "compact", "none", True),
    BenchmarkScenario(11, "Aesthetic", "Make this email sound more polite: 'Send me the files now.'", "compact", "none", True),
    BenchmarkScenario(12, "Aesthetic", "Fix the grammar in this sentence: 'He don't know nothing.'", "compact", "none", True),
    BenchmarkScenario(13, "Aesthetic", "Rephrase this sentence to sound more casual: 'I shall attend the gathering.'", "compact", "none", True),
    BenchmarkScenario(14, "Aesthetic", "Make this bulleted list concise and punchy.", "compact", "none", True),
    BenchmarkScenario(15, "Aesthetic", "Format the following text as a markdown table.", "compact", "none", True),
    BenchmarkScenario(16, "Aesthetic", "Proofread this paragraph for typos and spelling mistakes.", "compact", "none", True),
    BenchmarkScenario(17, "Aesthetic", "Make this tweet shorter than 100 characters.", "compact", "none", True),
    BenchmarkScenario(18, "Aesthetic", "Translate 'Where is the library?' into Spanish.", "compact", "none", True),
    BenchmarkScenario(19, "Aesthetic", "Translate 'Thank you very much' into French.", "compact", "none", True),
    BenchmarkScenario(20, "Aesthetic", "Make this product description sound more exciting and friendly.", "compact", "none", True),

    # --- Category 2: Low Logic, High Knowledge (History, Humanities, Law, Medicine) [16 prompts] ---
    BenchmarkScenario(21, "Deep Knowledge", "Explain the historical significance of the Peace of Westphalia in 1648 and its impact on modern nation-state sovereignty.", "mid", "none", False),
    BenchmarkScenario(22, "Deep Knowledge", "Compare the epistemology of John Locke's empiricism with Immanuel Kant's transcendental idealism.", "mid", "none", False),
    BenchmarkScenario(23, "Deep Knowledge", "Summarize the legal doctrine of substantive due process in United States constitutional jurisprudence.", "mid", "none", False),
    BenchmarkScenario(24, "Deep Knowledge", "What were the primary economic mechanisms causing hyperinflation during the Weimar Republic?", "mid", "none", False),
    BenchmarkScenario(25, "Deep Knowledge", "Analyze the literary symbolism of the green light in F. Scott Fitzgerald's The Great Gatsby.", "mid", "none", False),
    BenchmarkScenario(26, "Deep Knowledge", "Describe the physiological role of the renin-angiotensin-aldosterone system (RAAS) in arterial blood pressure regulation.", "mid", "none", False),
    BenchmarkScenario(27, "Deep Knowledge", "Examine the geopolitical causes and consequences of the 1973 OPEC oil embargo.", "mid", "none", False),
    BenchmarkScenario(28, "Deep Knowledge", "Discuss the linguistic shift from Old English to Middle English following the Norman Conquest of 1066.", "mid", "none", False),
    BenchmarkScenario(29, "Deep Knowledge", "Explain the philosophical difference between act utilitarianism and rule utilitarianism.", "mid", "none", False),
    BenchmarkScenario(30, "Deep Knowledge", "Detail the mechanism of action of monoclonal antibodies in modern oncology immunotherapy.", "mid", "none", False),
    BenchmarkScenario(31, "Deep Knowledge", "Outline the architectural evolution of Gothic cathedrals from Romanesque basilicas.", "mid", "none", False),
    BenchmarkScenario(32, "Deep Knowledge", "Explain the differences between civil law legal traditions and common law judicial precedent.", "mid", "none", False),
    BenchmarkScenario(33, "Deep Knowledge", "Describe the role of transcription factors and epigenetic histone methylation in embryonic stem cell differentiation.", "mid", "none", False),
    BenchmarkScenario(34, "Deep Knowledge", "Analyze the sociological concept of habitus formulated by Pierre Bourdieu.", "mid", "none", False),
    BenchmarkScenario(35, "Deep Knowledge", "Discuss the macroeconomic tradeoffs between quantitative easing and interest rate hikes during inflationary cycles.", "mid", "none", False),
    BenchmarkScenario(36, "Deep Knowledge", "What was the role of the Silk Road in transmitting artistic and religious motifs between the Hellenistic world and East Asia?", "mid", "none", False),

    # --- Category 3: Medium Logic, Low Knowledge (String Formatting, Parsing, Elementary Logic) [16 prompts] ---
    BenchmarkScenario(37, "Basic Logic", "Write a python function to parse a CSV string and return a list of dictionaries.", "compact", "low", False),
    BenchmarkScenario(38, "Basic Logic", "Given an array of strings, filter out all strings that have less than 5 characters.", "compact", "low", False),
    BenchmarkScenario(39, "Basic Logic", "Convert a temperature value from Celsius to Fahrenheit and Kelvin in Python.", "compact", "low", False),
    BenchmarkScenario(40, "Basic Logic", "Write a function to check if a given word is a palindrome, ignoring casing.", "compact", "low", False),
    BenchmarkScenario(41, "Basic Logic", "How do I split a string by commas but ignore commas inside quoted strings in Python?", "compact", "low", False),
    BenchmarkScenario(42, "Basic Logic", "Write a SQL query to find all users whose signup date is within the last 30 days.", "compact", "low", False),
    BenchmarkScenario(43, "Basic Logic", "Create a function to calculate the arithmetic mean and standard deviation of a numeric list.", "compact", "low", False),
    BenchmarkScenario(44, "Basic Logic", "Write a function that counts vowel occurrences in a given text string.", "compact", "low", False),
    BenchmarkScenario(45, "Basic Logic", "How do I safely parse a timestamp string into an ISO-8601 UTC datetime in TypeScript?", "compact", "low", False),
    BenchmarkScenario(46, "Basic Logic", "Given a dictionary, invert its keys and values assuming all values are unique.", "compact", "low", False),
    BenchmarkScenario(47, "Basic Logic", "Write a bash one-liner to find all files ending in .log older than 7 days and delete them.", "compact", "low", False),
    BenchmarkScenario(48, "Basic Logic", "Implement a function to generate all two-element combinations from a list of numbers.", "compact", "low", False),
    BenchmarkScenario(49, "Basic Logic", "Write a regex to validate an IPv4 address string.", "mid", "medium", False),
    BenchmarkScenario(50, "Basic Logic", "Write a function that checks if two strings are anagrams of each other.", "compact", "low", False),
    BenchmarkScenario(51, "Basic Logic", "How to extract query parameters from a URL string in JavaScript without external libraries?", "compact", "low", False),
    BenchmarkScenario(52, "Basic Logic", "Generate a list of prime numbers up to 100 using Python.", "compact", "low", False),

    # --- Category 4: Medium Logic, High Knowledge (Architecture & Engineering Tradeoffs) [16 prompts] ---
    BenchmarkScenario(53, "System Tradeoffs", "Compare database indexing strategies: B-Tree vs LSM-Tree in write-heavy transactional workloads.", "mid", "medium", False),
    BenchmarkScenario(54, "System Tradeoffs", "Evaluate the tradeoffs between event sourcing and CRUD state persistence for a fintech ledger service.", "mid", "medium", False),
    BenchmarkScenario(55, "System Tradeoffs", "Design a rate limiter algorithm: Token Bucket vs Leaky Bucket vs Sliding Window Counter. Analyze memory vs accuracy.", "mid", "medium", False),
    BenchmarkScenario(56, "System Tradeoffs", "How should we design a zero-downtime database migration schema for Postgres tables with 100M rows?", "mid", "medium", False),
    BenchmarkScenario(57, "System Tradeoffs", "Compare GraphQL vs gRPC vs REST for an internal microservice service mesh handling 50k QPS.", "mid", "medium", False),
    BenchmarkScenario(58, "System Tradeoffs", "Analyze cache invalidation strategies: Write-Through vs Write-Back vs Cache-Aside under high concurrency.", "mid", "medium", False),
    BenchmarkScenario(59, "System Tradeoffs", "How do OAuth2 Authorization Code Flow with PKCE mitigate authorization code interception attacks on mobile apps?", "mid", "medium", False),
    BenchmarkScenario(60, "System Tradeoffs", "Compare Kafka vs RabbitMQ message brokers: Partition ordering guarantees, consumer groups, and persistence models.", "mid", "medium", False),
    BenchmarkScenario(61, "System Tradeoffs", "Architect an idempotent webhook ingestion pipeline in Go with dead letter queue retries and deduplication.", "mid", "medium", False),
    BenchmarkScenario(62, "System Tradeoffs", "Explain database isolation levels: Read Committed vs Repeatable Read vs Serializable, and phantom read anomalies.", "mid", "medium", False),
    BenchmarkScenario(63, "System Tradeoffs", "Design a consistent hashing ring algorithm with virtual nodes to distribute cache keys across dynamic cluster nodes.", "mid", "medium", False),
    BenchmarkScenario(64, "System Tradeoffs", "Discuss the CAP theorem implications on DynamoDB vs Google Cloud Spanner global replication models.", "mid", "medium", False),
    BenchmarkScenario(65, "System Tradeoffs", "What are the performance implications of TCP head-of-line blocking in HTTP/2 compared to QUIC and HTTP/3?", "mid", "medium", False),
    BenchmarkScenario(66, "System Tradeoffs", "Compare Kubernetes horizontal pod autoscaling metrics: CPU/Memory utilization vs custom Prometheus QPS metrics.", "mid", "medium", False),
    BenchmarkScenario(67, "System Tradeoffs", "How do modern garbage collectors (like Go's concurrent collector or Java ZGC) minimize stop-the-world pauses?", "mid", "medium", False),
    BenchmarkScenario(68, "System Tradeoffs", "Design a distributed tracing schema adhering to OpenTelemetry standards with traceparent propagation.", "mid", "medium", False),

    # --- Category 5: High Logic, Low Knowledge (Algorithmic Logic, DP, Puzzles, Proofs) [16 prompts] ---
    BenchmarkScenario(69, "Algorithm", "Implement a dynamic programming algorithm to find the longest common subsequence of two strings. Prove time complexity O(n*m).", "mid", "high", False),
    BenchmarkScenario(70, "Algorithm", "Solve the 0/1 knapsack problem with dynamic programming and memory optimization to O(W) space.", "mid", "high", False),
    BenchmarkScenario(71, "Algorithm", "Write a function to verify if a Sudoku board is valid according to strict rules in O(1) extra space.", "mid", "medium", False),
    BenchmarkScenario(72, "Algorithm", "Implement Dijkstra's shortest path algorithm using a min-heap priority queue in Python. Provide complexity analysis.", "mid", "high", False),
    BenchmarkScenario(73, "Algorithm", "Construct a regular expression matching valid floating point numbers with scientific notation, without catastrophic backtracking.", "mid", "high", False),
    BenchmarkScenario(74, "Algorithm", "Prove by mathematical induction that 1 + 2 + ... + n = n(n+1)/2 for all positive integers n.", "mid", "high", False),
    BenchmarkScenario(75, "Algorithm", "Implement an algorithm for topological sorting of a Directed Acyclic Graph (DAG) using Kahn's algorithm.", "mid", "high", False),
    BenchmarkScenario(76, "Algorithm", "Solve the LeetCode problem 'Trapping Rain Water' in O(n) time and O(1) auxiliary space using two pointers.", "mid", "high", False),
    BenchmarkScenario(77, "Algorithm", "Write a recursive descent parser for arithmetic expressions with operator precedence (*, /, +, -) and parentheses.", "mid", "high", False),
    BenchmarkScenario(78, "Algorithm", "Implement the Boyer-Moore majority vote algorithm and mathematically prove why it works in O(n) time.", "mid", "high", False),
    BenchmarkScenario(79, "Algorithm", "Given an array of integers, find the contiguous subarray which has the largest sum using Kadane's algorithm.", "mid", "medium", False),
    BenchmarkScenario(80, "Algorithm", "Implement an LRU Cache with O(1) get and put operations using a doubly linked list and hash map.", "mid", "high", False),
    BenchmarkScenario(81, "Algorithm", "Solve the N-Queens problem using backtracking with bitmask optimization to count all valid placements.", "mid", "high", False),
    BenchmarkScenario(82, "Algorithm", "Construct a regex that matches balanced nested parentheses up to 3 levels deep.", "mid", "high", False),
    BenchmarkScenario(83, "Algorithm", "Implement binary search on a rotated sorted array in O(log n) time, handling duplicate elements.", "mid", "high", False),
    BenchmarkScenario(84, "Algorithm", "Prove that any comparison-based sorting algorithm has an asymptotic lower bound of Omega(n log n).", "mid", "high", False),

    # --- Category 6: High Logic, High Knowledge (Distributed Systems, Concurrency, Security) [16 prompts] ---
    BenchmarkScenario(85, "Distributed Rigor", "Debug a concurrent race condition in a Go distributed mutex where lock leasing and split-brain cause data corruption under network partitions.", "frontier", "high", False),
    BenchmarkScenario(86, "Distributed Rigor", "Formally verify safety and liveness invariants for the Raft consensus algorithm during leader election under asynchronous message delays.", "frontier", "high", False),
    BenchmarkScenario(87, "Distributed Rigor", "Analyze a deadlock scenario in PostgreSQL when two concurrent transactions update rows in reverse order under Serializable Snapshot Isolation.", "frontier", "high", False),
    BenchmarkScenario(88, "Distributed Rigor", "How does Paxos guarantee consensus under partial network partitions, and how does Multi-Paxos optimize phase 1 prepare rounds?", "frontier", "high", False),
    BenchmarkScenario(89, "Distributed Rigor", "Verify the cryptographic security proof of the Diffie-Hellman key exchange against man-in-the-middle attacks using digital signatures.", "frontier", "high", False),
    BenchmarkScenario(90, "Distributed Rigor", "Implement a thread-safe lock-free concurrent queue using atomic compare-and-swap (CAS) and explain how to prevent the ABA problem.", "frontier", "high", False),
    BenchmarkScenario(91, "Distributed Rigor", "Analyze the Raft log compaction mechanism and how snapshotting interacts with stale follower replication RPCs without race conditions.", "frontier", "high", False),
    BenchmarkScenario(92, "Distributed Rigor", "Prove the correctness of Lamport logical timestamps in establishing a partial order of events in distributed systems.", "frontier", "high", False),
    BenchmarkScenario(93, "Distributed Rigor", "Debug a memory leak in a Rust async runtime caused by circular Arc references in futures and Tokio task spawning.", "frontier", "high", False),
    BenchmarkScenario(94, "Distributed Rigor", "Formulate a formal proof that Byzantine Agreement is impossible with 3 nodes if 1 is faulty, but solvable with 4 nodes.", "frontier", "high", False),
    BenchmarkScenario(95, "Distributed Rigor", "Explain how Google Spanner uses TrueTime atomic clocks to provide external consistency without cross-datacenter locking.", "frontier", "high", False),
    BenchmarkScenario(96, "Distributed Rigor", "Design an algorithm for conflict-free replicated data types (CRDTs): State-based OR-Set with causal deletion guarantees.", "frontier", "high", False),
    BenchmarkScenario(97, "Distributed Rigor", "Identify timing side-channel vulnerabilities in constant-time cryptographic modular exponentiation implementations.", "frontier", "high", False),
    BenchmarkScenario(98, "Distributed Rigor", "Analyze deadlock detection and wait-for graph cycle resolution in distributed transactions with Two-Phase Locking.", "frontier", "high", False),
    BenchmarkScenario(99, "Distributed Rigor", "Debug an issue where ephemeral TLS session renegotiation causes thread-safety race conditions in an HTTP/2 connection pool.", "frontier", "high", False),
    BenchmarkScenario(100, "Distributed Rigor", "Formally prove that Vector Clocks capture the causal relationship between distributed events iff a -> b.", "frontier", "high", False),
]
