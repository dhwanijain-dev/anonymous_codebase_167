'use client'

import { createContext, useContext, useState, useCallback, useRef, type ReactNode } from 'react'

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000'

// =============================================================================
// TYPES
// =============================================================================

/**
 * BACKEND INTEGRATION: Chat History Item
 *
 * Represents a single chat in the user's history.
 * When connected to a backend, each chat should have a unique ID
 * so it can be fetched/resumed.
 *
 * API Endpoint: GET /api/chats
 * Response: { chats: ChatItem[] }
 */
export interface ChatItem {
  id: string
  title: string
  createdAt?: string
}

/**
 * BACKEND INTEGRATION: Analysis Result
 *
 * Represents the result returned after submitting a query.
 * The backend should return structured analysis data.
 *
 * API Endpoint: POST /api/analysis/query
 * Response: { result: AnalysisResult }
 */
export type Box = [number, number, number, number]

export type Detection = {
  box: Box
  label: string
  imageIndex?: number
}

export type AnalysisTask = {
  task: string
  result: {
    answer: string
    caption?: string | null
    boxes?: Box[]
    evidence_image?: string | null
    is_grounding?: boolean
    latency_seconds?: number
  }
  model: Record<string, unknown>
}

export type AnalysisResult = {
  request_id: string
  answer: string | null
  tasks: AnalysisTask[]
  detections?: Detection[]
  execution_trace: {
    total_latency_seconds: number
    selected_tasks: string[]
    workflow: string[]
  }
}

const DETECTION_PATTERN = /\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s+([^\]]+?)\s*\]/g

function parseDetections(value: string): Detection[] {
  return Array.from(value.matchAll(DETECTION_PATTERN), (match) => ({
    box: [
      Number(match[1]),
      Number(match[2]),
      Number(match[3]),
      Number(match[4]),
    ],
    label: match[5].trim(),
  }))
}

function getResponseAnswer(body: unknown): string {
  if (typeof body === 'string') return body
  if (Array.isArray(body)) return body.map(getResponseAnswer).filter(Boolean).join(' ')

  if (body && typeof body === 'object') {
    const record = body as Record<string, unknown>

    if ('answer' in record) {
      return String(record.answer ?? '')
    }

    if ('result' in record) {
      return getResponseAnswer(record.result)
    }
  }

  return ''
}

function parseBackendBody(text: string): unknown {
  try {
    return JSON.parse(text)
  } catch {
    const answerMatch = text.match(
      /["']answer["']\s*:\s*["']([\s\S]*?)["']\s*[,}]/
    )

    return answerMatch ? { answer: answerMatch[1] } : text
  }
}

export function normalizeAnalysisResponse(
  body: unknown,
  requestId: string
): AnalysisResult {
  const answer = getResponseAnswer(body)
  const detections = parseDetections(answer)
  const record = body && typeof body === 'object'
    ? body as Record<string, unknown>
    : {}
  const payload = record.result && typeof record.result === 'object'
    ? record.result as Record<string, unknown>
    : record
  const tasks = Array.isArray(payload.tasks)
    ? payload.tasks as AnalysisTask[]
    : []
  const taskDetections = tasks.flatMap((task) => {
    const boxes = (task.result?.boxes ?? []).map((box) => ({
      box,
      label: task.task || 'Detected object',
    }))
    const textDetections = parseDetections(task.result?.answer ?? '')

    return [...boxes, ...textDetections]
  })
  const allDetections = [...taskDetections, ...detections]
  const finalAnswer = allDetections.length > 0
    ? `Detected ${allDetections.length} ${allDetections.length === 1 ? 'object' : 'objects'}.`
    : answer

  return {
    request_id:
      typeof record.request_id === 'string'
        ? record.request_id
        : requestId,
    answer: finalAnswer,
    tasks,
    detections: allDetections,
    execution_trace:
      (payload.execution_trace as AnalysisResult['execution_trace']) ?? {
        total_latency_seconds: 0,
        selected_tasks: [],
        workflow: [],
      },
  }
}
/**
 * BACKEND INTEGRATION: Dataset
 *
 * Represents an available satellite dataset for analysis.
 *
 * API Endpoint: GET /api/datasets
 * Response: { datasets: Dataset[] }
 */
export interface Dataset {
  id: string
  name: string
  resolution: string
  description?: string
}

export type ChatMessage = {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: number
  images?: string[]
  result?: AnalysisResult
}

export type ChatRecord = {
  id: string
  title: string
  messages: ChatMessage[]
  datasetId?: string | null
  updatedAt: number
}

/**
 * BACKEND INTEGRATION: User Profile
 *
 * Current authenticated user info.
 *
 * API Endpoint: GET /api/auth/me
 * Response: { user: UserProfile }
 */
export interface UserProfile {
  id: string
  name: string
  initials: string
  email?: string
  avatarUrl?: string
}

// =============================================================================
// CONTEXT SHAPE
// =============================================================================

interface SatQueryContextValue {
  // --- Theme ---
  /** Current dark mode state. Persisted in localStorage. */
  darkMode: boolean
  /** Toggle dark/light mode */
  toggleDarkMode: () => void

  // --- Navigation ---
  /** Currently active navigation item in the left sidebar */
  activeNav: string
  /** Set the active navigation item */
  setActiveNav: (nav: string) => void
activeChat: ChatRecord | null
saveChatToRecents: (chat: ChatRecord) => void
startNewSession: () => void
  // --- Query / Chat ---
  /** Current text in the composer input */
  query: string
  /** Update the composer input text */
  setQuery: (q: string) => void
  /**
   * BACKEND INTEGRATION: Submit Query
   *
   * API Endpoint: POST /api/analysis/query
   * Request: { query: string, dataset: string, tool: string | null, imageId: string | null }
   * Response: { result: AnalysisResult }
   *
   * Replace the local state update with a fetch call to your backend.
   * Set `isAnalyzing` to true before the call, and false after.
   */
  submitQuery: () => Promise<void>
  /** The most recently submitted query text (for display) */
  sentQuery: string
  /** Whether an analysis is currently in progress */
  isAnalyzing: boolean
  /** Results from the most recent analysis */
  analysisResults: AnalysisResult | null

  // --- Dataset ---
  /** Currently selected dataset for analysis */
  dataset: Dataset
  /**
   * BACKEND INTEGRATION: Change Dataset
   *
   * API Endpoint: GET /api/datasets/:id
   * Response: { dataset: Dataset }
   *
   * When a user selects a dataset, fetch its full details from the backend.
   */
  setDataset: (ds: Dataset) => void
  /** List of available datasets */
  availableDatasets: Dataset[]

  // --- Tools ---
  /** Currently selected analysis tool (null = none) */
  selectedTool: string | null
  /** Set the active analysis tool */
  setSelectedTool: (tool: string | null) => void

  // --- Recent Chats ---
  /**
   * BACKEND INTEGRATION: Recent Chats
   *
   * API Endpoint: GET /api/chats?limit=5&sort=recent
   * Response: { chats: ChatItem[] }
   *
   * Fetch recent chats from the backend on mount or when the user
   * navigates to the chat section.
   */
  recentChats: ChatItem[]
  /**
   * BACKEND INTEGRATION: Load Chat
   *
   * API Endpoint: GET /api/chats/:id
   * Response: { chat: ChatItem, messages: Message[] }
   *
   * Load a specific chat's messages into the workspace.
   */
  loadChat: (chat: ChatItem) => void

  // --- Upload ---
  /**
   * BACKEND INTEGRATION: Upload Image
   *
   * API Endpoint: POST /api/images/upload
   * Request: FormData with file
   * Response: { imageId: string, url: string, metadata: object }
   *
   * Upload a satellite image for analysis.
   */
  uploadedImages: File[]
  lastSubmittedImages: File[]
  addUploadedImage: (file: File) => void
  removeUploadedImage: (index: number) => void

  // --- User ---
  /**
   * BACKEND INTEGRATION: User Profile
   *
   * API Endpoint: GET /api/auth/me
   * Response: { user: UserProfile }
   *
   * Fetch the current user's profile on app load.
   */
  user: UserProfile

  // --- Dialogs ---
  /** Whether the upload dialog is open */
  isUploadDialogOpen: boolean
  setUploadDialogOpen: (open: boolean) => void
  /** Whether the advanced options dialog is open */
  isAdvancedDialogOpen: boolean
  setAdvancedDialogOpen: (open: boolean) => void
  /** Whether the mobile sidebar sheet is open */
  isMobileSidebarOpen: boolean
  setMobileSidebarOpen: (open: boolean) => void
}

// =============================================================================
// DEFAULT DATA
// =============================================================================

/** Default datasets — replace with API fetch in production */
const DEFAULT_DATASETS: Dataset[] = [
  { id: 'sentinel-2', name: 'Sentinel-2', resolution: '10 m resolution', description: 'Multi-spectral imaging mission' },
  { id: 'landsat-9', name: 'Landsat 9', resolution: '30 m resolution', description: 'Operational Land Imager' },
  { id: 'modis', name: 'MODIS', resolution: '250 m resolution', description: 'Moderate Resolution Imaging' },
  { id: 'sentinel-1', name: 'Sentinel-1', resolution: '5 m resolution', description: 'SAR imaging mission' },
]

/** Default recent chats — replace with API fetch in production */
// const DEFAULT_RECENT_CHATS: ChatItem[] = [
//   { id: '1', title: 'Deforestation analysis ...' },
//   { id: '2', title: 'Flood extent in Assam' },
//   { id: '3', title: 'Crop health comparison' },
//   { id: '4', title: 'Urban expansion Delhi' },
//   { id: '5', title: 'Coastal change analysis' },
// ]

/** Default user — replace with auth integration in production */
const DEFAULT_USER: UserProfile = {
  id: '1',
  name: 'Dhwani',
  initials: 'DJ',
  email: 'dhwani@example.com',
}

// =============================================================================
// CONTEXT + PROVIDER
// =============================================================================

const SatQueryContext = createContext<SatQueryContextValue | null>(null)

/**
 * Hook to access the SatQuery app context.
 * Must be used within a <SatQueryProvider>.
 */
export function useSatQuery() {
  const ctx = useContext(SatQueryContext)
  if (!ctx) throw new Error('useSatQuery must be used within SatQueryProvider')
  return ctx
}


/**
 * BACKEND INTEGRATION: SatQueryProvider
 *
 * This provider initializes all app state. In production, replace the
 * default values with data fetched from your backend API on mount:
 *
 * - GET /api/auth/me → user
 * - GET /api/chats?limit=5&sort=recent → recentChats
 * - GET /api/datasets → availableDatasets
 *
 * Use useEffect hooks to fetch this data on component mount.
 */
export function SatQueryProvider({ children }: { children: ReactNode }) {
  // --- Theme ---
  const [darkMode, setDarkMode] = useState(false)
  const toggleDarkMode = useCallback(() => setDarkMode((prev) => !prev), [])

  // --- Navigation ---
  const [activeNav, setActiveNav] = useState('New Chat')

  // --- Query ---
  const [query, setQuery] = useState('')
  const [sentQuery, setSentQuery] = useState('')
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [analysisResults, setAnalysisResults] = useState<AnalysisResult | null>(null)
  const [uploadedImages, setUploadedImages] = useState<File[]>([])
  const [lastSubmittedImages, setLastSubmittedImages] = useState<File[]>([])
  const requestInFlightRef = useRef(false)
 
  /**
   * BACKEND INTEGRATION: submitQuery
   *
   * Replace the setTimeout mock below with an actual API call:
   *
   * async function submitQuery() {
   *   if (!query.trim()) return
   *   const q = query.trim()
   *   setSentQuery(q)
   *   setQuery('')
   *   setIsAnalyzing(true)
   *   try {
   *     const res = await fetch('/api/analysis/query', {
   *       method: 'POST',
   *       headers: { 'Content-Type': 'application/json' },
   *       body: JSON.stringify({
   *         query: q,
   *         datasetId: dataset.id,
   *         tool: selectedTool,
   *         imageIds: uploadedImages.map(f => f.name), // replace with actual uploaded IDs
   *       }),
   *     })
   *     const data = await res.json()
   *     setAnalysisResults(data.result)
   *   } catch (error) {
   *     console.error('Analysis failed:', error)
   *   } finally {
   *     setIsAnalyzing(false)
   *   }
   * }
   */
  // const submitQuery = useCallback(() => {
  //   if (!query.trim()) return
  //   const q = query.trim()
  //   setSentQuery(q)
  //   setQuery('')
  //   setIsAnalyzing(true)

  //   // Mock: simulate a 2-second analysis delay
  //   setTimeout(() => {
  //     setAnalysisResults({
  //       id: Date.now().toString(),
  //       query: q,
  //       summary: `Analysis complete for: "${q}"`,
  //       createdAt: new Date().toISOString(),
  //     })
  //     setIsAnalyzing(false)
  //   }, 2000)
  // }, [query])

const submitQuery = useCallback(async () => {
  if (
    requestInFlightRef.current ||
    !query.trim() ||
    isAnalyzing ||
    uploadedImages.length === 0
  ) return

  const currentPrompt = query.trim()
  const currentFiles = [...uploadedImages]

  requestInFlightRef.current = true
  setLastSubmittedImages(currentFiles)
  setSentQuery(currentPrompt)
  setQuery('')
  setIsAnalyzing(true)

  try {
    const form = new FormData()

    form.append('query', currentPrompt)

    form.append(
      'modalities',
      currentFiles.length === 2 ? 'unknown,pair' : 'unknown'
    )

    currentFiles.forEach((file, index) => {
      form.append(
        index === 0 ? 'image1' : 'image2',
        file,
        file.name
      )
    })

    const response = await fetch(`${API_BASE_URL}/analyze`, {
      method: 'POST',
      body: form,
    })

    const body = parseBackendBody(await response.text())

    if (!response.ok) {
      const errorBody = body as { detail?: string }
      throw new Error(
        errorBody.detail ?? 'The analysis could not be completed.'
      )
    }

    setAnalysisResults(
      normalizeAnalysisResponse(body, `request-${Date.now()}`)
    )
    setUploadedImages([])
  } catch (error) {
    console.error('SatQuery analysis failed:', error)

    setAnalysisResults(null)
  } finally {
    requestInFlightRef.current = false
    setIsAnalyzing(false)
  }
}, [query, uploadedImages, isAnalyzing])

  // --- Dataset ---
  const [dataset, setDataset] = useState<Dataset>(DEFAULT_DATASETS[0])
  const [availableDatasets] = useState<Dataset[]>(DEFAULT_DATASETS)

  // --- Tools ---
  const [selectedTool, setSelectedTool] = useState<string | null>(null)

  // --- Recent Chats ---
const [recentChats, setRecentChats] = useState<ChatRecord[]>([])
const [activeChat, setActiveChat] = useState<ChatRecord | null>(null)
/**
   * BACKEND INTEGRATION: loadChat
   *
   * API Endpoint: GET /api/chats/:id
   * Response: { chat: ChatItem, messages: Message[] }
   *
   * Load full chat history and display in workspace.
   */
 const loadChat = useCallback((chat: ChatRecord) => {
  setActiveNav('New Chat')
  setActiveChat(chat)

  setQuery('')
  setSentQuery('')
  setAnalysisResults(null)
}, [])
  const saveChatToRecents = useCallback((chat: ChatRecord) => {
  setRecentChats((previous) => {
    const filtered = previous.filter((item) => item.id !== chat.id)

    return [chat, ...filtered].slice(0, 20)
  })
}, [])

const startNewSession = useCallback(() => {
  setActiveChat(null)
  setQuery('')
  setSentQuery('')
  setAnalysisResults(null)
  setUploadedImages([])
}, [])

  // --- Upload ---

  /**
  const [uploadedImages, setUploadedImages] = useState<File[]>([])
   * BACKEND INTEGRATION: addUploadedImage
   *
   * API Endpoint: POST /api/images/upload
   * Request: FormData with file
   * Response: { imageId: string, url: string }
   *
   * Upload the file to your backend and store the returned imageId.
   */
 const addUploadedImage = useCallback((file: File) => {
  setUploadedImages((previous) => {
    if (previous.length >= 2) {
      return previous
    }

    return [...previous, file]
  })
}, [])

  const removeUploadedImage = useCallback((index: number) => {
    setUploadedImages((prev) => prev.filter((_, i) => i !== index))
  }, [])

  // --- User ---
  const [user] = useState<UserProfile>(DEFAULT_USER)

  // --- Dialogs ---
  const [isUploadDialogOpen, setUploadDialogOpen] = useState(false)
  const [isAdvancedDialogOpen, setAdvancedDialogOpen] = useState(false)
  const [isMobileSidebarOpen, setMobileSidebarOpen] = useState(false)

  const value: SatQueryContextValue = {
    darkMode,
    toggleDarkMode,
    activeNav,
    setActiveNav,
    query,
    setQuery,
    submitQuery,
    sentQuery,
    isAnalyzing,
    analysisResults,
    dataset,
    setDataset,
    availableDatasets,
    selectedTool,
    setSelectedTool,
    recentChats,
    activeChat,
    saveChatToRecents,
    startNewSession,
    loadChat,
    uploadedImages,
    lastSubmittedImages,
    addUploadedImage,
    removeUploadedImage,
    user,
    isUploadDialogOpen,
    setUploadDialogOpen,
    isAdvancedDialogOpen,
    setAdvancedDialogOpen,
    isMobileSidebarOpen,
    setMobileSidebarOpen,
  }

  return <SatQueryContext.Provider value={value}>{children}</SatQueryContext.Provider>
}
