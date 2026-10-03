'use client'

import { useEffect, useRef, useState } from 'react'
import { Plus } from 'lucide-react'

import { Button } from '@/components/ui/button'
import { useSatQuery } from '@/components/satquery-context'

import { HeroSection } from './hero-section'
import { Composer } from './composer'
import { ChatThread, type ChatMessage } from './ChatThread'

export function ChatShell() {
  const {
    sentQuery,
    isAnalyzing,
    analysisResults,

    setQuery,

    lastSubmittedImages,

    activeChat,
    saveChatToRecents,
    startNewSession,
  } = useSatQuery()

  const [messages, setMessages] = useState<ChatMessage[]>([])

  const chatIdRef = useRef(`chat-${Date.now()}`)

  const lastSentQueryRef = useRef<string | null>(null)
  const lastResultRef = useRef<string | null>(null)
  const submittedImageUrlsRef = useRef<string[]>([])

  useEffect(() => {
    return () => {
      submittedImageUrlsRef.current.forEach((url) => URL.revokeObjectURL(url))
    }
  }, [])

  /*
   * Load an existing conversation.
   */
  useEffect(() => {
    if (!activeChat) return

    setMessages(activeChat.messages)
    chatIdRef.current = activeChat.id

    lastSentQueryRef.current = null
    lastResultRef.current = null
  }, [activeChat])

  /*
   * Add user's query to the visible conversation.
   */
  useEffect(() => {
    if (!sentQuery) return

    if (sentQuery === lastSentQueryRef.current) {
      return
    }

    lastSentQueryRef.current = sentQuery

    submittedImageUrlsRef.current.forEach((url) => URL.revokeObjectURL(url))
    const previews = lastSubmittedImages.map((file) =>
      URL.createObjectURL(file)
    )
    submittedImageUrlsRef.current = previews

    setMessages((previous) => [
      ...previous,
      {
        id: `user-${Date.now()}`,
        role: 'user',
        content: sentQuery,
        images: previews,
        timestamp: Date.now(),
      },
    ])
  }, [sentQuery, lastSubmittedImages])

  /*
   * Add backend response.
   */
  useEffect(() => {
    if (!analysisResults || isAnalyzing) return

    if (analysisResults.request_id === lastResultRef.current) {
      return
    }

    lastResultRef.current = analysisResults.request_id

    const assistantMessage: ChatMessage = {
      id: `assistant-${Date.now()}`,
      role: 'assistant',
      content:
        analysisResults.answer ??
        analysisResults.tasks?.[0]?.result?.answer ??
        'No answer was returned.',
      timestamp: Date.now(),
      images: [...submittedImageUrlsRef.current],
      result: analysisResults,
    }

    setMessages((previous) => [...previous, assistantMessage])
  }, [analysisResults, isAnalyzing])

  const hasStartedChat = messages.length > 0

  function handleNewChat() {
    if (messages.length > 0) {
      saveChatToRecents({
        id: chatIdRef.current,
        title:
          messages[0]?.content.slice(0, 60) ??
          'Untitled analysis',
        messages,
        updatedAt: Date.now(),
      })
    }

    setMessages([])

    chatIdRef.current = `chat-${Date.now()}`

    lastSentQueryRef.current = null
    lastResultRef.current = null

    setQuery('')

    startNewSession()
  }

  return (
    <div className="chat-shell">
      {hasStartedChat && (
        <div className="chat-shell-topbar">
          <Button
            variant="ghost"
            size="sm"
            onClick={handleNewChat}
            className="new-chat-btn"
          >
            <Plus className="mr-1.5 h-4 w-4" />
            New chat
          </Button>
        </div>
      )}

      <div className="chat-shell-body">
        {hasStartedChat ? (
          <ChatThread
            messages={messages}
            isAnalyzing={isAnalyzing}
          />
        ) : (
          <HeroSection />
        )}
      </div>

      <div className="chat-shell-composer">
        <Composer />
      </div>

      <style jsx>{`
        .chat-shell {
          display: flex;
          flex-direction: column;
          height: 100%;
          min-height: 0;
        }

        .chat-shell-topbar {
          display: flex;
          justify-content: flex-end;
          padding: 8px 16px;
          animation: shellFadeIn 200ms ease-out;
        }

        .chat-shell-body {
          flex: 1;
          min-height: 0;
          overflow-y: auto;
        }

        .chat-shell-composer {
          flex-shrink: 0;
          padding: 12px 16px 20px;
        }

        @keyframes shellFadeIn {
          from {
            opacity: 0;
            transform: translateY(-4px);
          }

          to {
            opacity: 1;
            transform: translateY(0);
          }
        }

        @media (prefers-reduced-motion: reduce) {
          .chat-shell-topbar {
            animation: none !important;
          }
        }
      `}</style>
    </div>
  )
}