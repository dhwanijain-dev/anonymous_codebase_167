'use client'
import { useState, useRef, useEffect, useMemo } from 'react'
import { Image as ImageIcon, ArrowUp, Settings2, Upload, Scan, Database, X, Loader2, FileImage, Check, Sparkles } from 'lucide-react'
import { useSatQuery } from '@/components/satquery-context'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Badge } from '@/components/ui/badge'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
  DialogClose
} from '@/components/ui/dialog'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger
} from '@/components/ui/dropdown-menu'

export function Composer() {
  const {
    query,
    setQuery,
    submitQuery,
    isAnalyzing,
    dataset,
    setDataset,
    availableDatasets,
    // NOTE: isUploadDialogOpen / setUploadDialogOpen are still exposed by the
    // context for backward compatibility, but this component no longer routes
    // attachment through a confirmation dialog (see handleAttachClick below).
    isAdvancedDialogOpen,
    setAdvancedDialogOpen,
    uploadedImages,
    addUploadedImage,
    removeUploadedImage
  } = useSatQuery()
const [cloudCover, setCloudCover] = useState(20)
const [dateFrom, setDateFrom] = useState('')
const [dateTo, setDateTo] = useState('')
const [selectedBands, setSelectedBands] = useState<string[]>([])
const [resolution, setResolution] = useState('')
const [error, setError] = useState('')

  const fileInputRef = useRef<HTMLInputElement>(null)

  // ---------------------------------------------------------------------
  // Inline attachment previews (Claude / ChatGPT style)
  // ---------------------------------------------------------------------
  // We derive a local objectURL per uploaded file so thumbnails can render
  // immediately, client-side, with no round trip to the backend. When the
  // real upload endpoint (POST /api/images/upload) responds with a stable
  // remote URL + imageId, swap the objectURL below for that remote URL and
  // keep the imageId to send along with the analysis request.
  const previewUrls = useMemo(
    () => uploadedImages.map((file) => URL.createObjectURL(file)),
    [uploadedImages]
  )

  useEffect(() => {
    // Revoke old object URLs whenever the file list changes / unmounts,
    // otherwise we leak memory as the user attaches/removes many images.
    return () => {
      previewUrls.forEach((url) => URL.revokeObjectURL(url))
    }
  }, [previewUrls])

  useEffect(() => {
    if (uploadedImages.length === 0 && fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }, [uploadedImages.length])

  // Newly-added thumbnails get a quick fade/scale-in so attaching an image
  // feels responsive rather than just "popping" into place.
  const [justAddedIndex, setJustAddedIndex] = useState<number | null>(null)
  const handleAttachClick = () => {
    if (uploadedImages.length >= 2) return

    fileInputRef.current?.click()
  }
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files) return

    const selectedFiles = Array.from(e.target.files)

    const supported = selectedFiles.filter((file) =>
      /\.(tif|tiff|png|jpe?g)$/i.test(file.name)
    )

    const remainingSlots = Math.max(0, 2 - uploadedImages.length)

    const filesToAdd = supported.slice(0, remainingSlots)

    filesToAdd.forEach((file) => {
      addUploadedImage(file)
    })

    if (filesToAdd.length > 0) {
      setJustAddedIndex(
        uploadedImages.length + filesToAdd.length - 1
      )
    }

    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  // Clicking the image icon (or the "Upload Image" action button) now opens
  // the native OS file picker directly — no intermediate confirmation
  // dialog. Selected files are attached immediately and rendered as
  // preview chips above the composer input, mirroring the Claude / ChatGPT
  // attach flow.

  const toggleBand = (band: string) => {
    setSelectedBands((prev) =>
      prev.includes(band) ? prev.filter(b => b !== band) : [...prev, band]
    )
  }

  const allBands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8']

  // ---------------------------------------------------------------------
  // Send handler
  // ---------------------------------------------------------------------
  const handleSubmit = () => {
    if (isAnalyzing) return

  if (!query.trim()) return

  if (uploadedImages.length === 0) {
    setError('Attach an image before sending your question.')
    return
  }

  setError('')

  void submitQuery()


    /**
     * BACKEND INTEGRATION: Recent Chats persistence
     * -----------------------------------------------------------------
     * This is the natural hook point for saving a chat/session entry so it
     * shows up in a "Recent" sidebar/list, the same way Claude/ChatGPT do.
     *
     * Suggested flow:
     * 1. On first submit in a session, generate a stable chatId (uuid) and
     *    keep it in context (e.g. SatQueryContext.currentChatId) so every
     *    subsequent message in the same session upserts the same record
     *    instead of creating a new "recent chat" per message.
     * 2. Build a lightweight record as soon as the query is sent (optimistic),
     *    then patch it once the analysis result comes back:
     *      {
     *        id: chatId,
     *        title: query.slice(0, 60),      // first message becomes the title
     *        lastMessage: query,
     *        datasetId: dataset?.id ?? null,
     *        imageNames: uploadedImages.map((f) => f.name),
     *        status: 'pending' | 'complete' | 'error',
     *        updatedAt: Date.now(),
     *      }
     * 3. Persist via POST /api/chats (create) then PATCH /api/chats/:id
     *    (update with results) — or, for a local-first/offline mode, mirror
     *    the same record into localStorage under a `recentChats` key and
     *    keep only the most recent N (e.g. 20), most-recent-first.
     * 4. Push the record into whatever shared store renders the sidebar
     *    (React context, Zustand, Redux, SWR/React Query cache) so the
     *    "Recent" list updates immediately without a refetch.
     * 5. On submitQuery's success/failure inside SatQueryContext, flip
     *    status to 'complete' | 'error' and store analysisResults.summary
     *    as a short preview snippet for the recent-chats list item.
     */

  }

  return (
    <>
      {/* -----------------------------------------------------------------
          Attachment preview strip — shown above the composer input bar,
          only when at least one image has been attached. Each chip fades
          + scales in on mount and can be removed individually.
      ------------------------------------------------------------------ */}
      {uploadedImages.length > 0 && (
        <div className="attachment-strip" role="list" aria-label="Attached images">
          {uploadedImages.map((file, i) => (
            <div
              key={`${file.name}-${file.lastModified}-${i}`}
              role="listitem"
              className="attachment-chip"
              style={{
                animation: i === justAddedIndex
                  ? 'attachmentIn 220ms cubic-bezier(0.16, 1, 0.3, 1)'
                  : undefined
              }}
            >
              <img src={previewUrls[i]} alt={file.name} className="attachment-chip-img" />
              <button
                type="button"
                aria-label={`Remove ${file.name}`}
                className="attachment-chip-remove"
                onClick={() => removeUploadedImage(i)}
              >
                <X className="h-3 w-3" />
              </button>
<span className="text-[11px] text-[#91a098]">
  {uploadedImages.length >= 2
    ? 'Maximum 2 images attached'
    : 'Attach up to 2 satellite images'}
</span>            
</div>
          ))}
        </div>
      )}

      {/* Composer input bar */}
     {/* ```tsx */}
      {/* Composer input bar */}
      <div className="composer">

        {/* Attach image */}
        <button
          type="button"
          aria-label="Attach image"
          onClick={handleAttachClick}
          disabled={uploadedImages.length >= 2}
          className="relative"
        >
          <ImageIcon />

          {uploadedImages.length > 0 && (
            <Badge className="absolute -top-2 -right-2 px-1 py-0 text-[10px] min-w-[16px] h-4 flex items-center justify-center">
              {uploadedImages.length}
            </Badge>
          )}
        </button>

        {/* Hidden native file input */}
        <input
          type="file"
          ref={fileInputRef}
          className="hidden"
          multiple
          accept=".tif,.tiff,.png,.jpg,.jpeg"
          onChange={handleFileChange}
        />

        {/* Question input */}
        <input
          aria-label="Question about satellite image"
          value={query}
          onChange={(e) => {
            setQuery(e.target.value)

            if (error) {
              setError('')
            }
          }}
          onKeyDown={(e) => {
            if (
              e.key === 'Enter' &&
              !e.nativeEvent.isComposing &&
              e.keyCode !== 229
            ) {
              e.preventDefault()
              handleSubmit()
            }
          }}
          placeholder="Type your question about the satellite image..."
        />

        {/* Advanced settings */}
        <Button
          variant="ghost"
          size="icon"
          aria-label="Advanced settings"
          onClick={() => setAdvancedDialogOpen(true)}
        >
          <Settings2 />
        </Button>

        {/* Send button */}
        <Button
          className={`send-button${
            isAnalyzing ? ' send-button-generating' : ''
          }`}
          size="icon"
          onClick={handleSubmit}
          aria-label={
            isAnalyzing
              ? 'Generating response'
              : 'Send question'
          }
          disabled={
            isAnalyzing ||
            !query.trim() ||
            uploadedImages.length === 0
          }
        >
          {isAnalyzing ? (
            <Loader2 className="h-4 w-4 animate-spin" />
          ) : (
            <ArrowUp className="send-button-icon" />
          )}
        </Button>
      </div>

      {/* Error */}
      {error && (
        <p className="mt-2 rounded-lg border border-[#e4b8a9] bg-[#fff2ed] px-3 py-2 text-xs text-[#a04b3d]">
          {error}
        </p>
      )}

      {/* Action buttons below composer */}
      <div className="composer-actions">

        {/* Upload image */}
        <Button
          variant="outline"
          onClick={handleAttachClick}
          disabled={uploadedImages.length >= 2}
        >
          <Upload
            data-icon="inline-start"
            className="mr-2 h-4 w-4"
          />
          Upload Image
        </Button>

        {/* Select area */}
        <Button variant="outline">
          <Scan
            data-icon="inline-start"
            className="mr-2 h-4 w-4"
          />
          Select Area
        </Button>

        {/* Dataset */}
        <DropdownMenu>
          <DropdownMenuTrigger className="inline-flex items-center justify-center rounded-md border px-4 py-2">
            <Database
              data-icon="inline-start"
              className="mr-2 h-4 w-4"
            />
            {dataset ? dataset.name : 'Choose Dataset'}
          </DropdownMenuTrigger>

          <DropdownMenuContent>
            {availableDatasets.map((ds) => (
              <DropdownMenuItem
                key={ds.id}
                onClick={() => setDataset(ds)}
              >
                {ds.name}
              </DropdownMenuItem>
            ))}
          </DropdownMenuContent>
        </DropdownMenu>

        {/* Advanced options */}
        <Button
          variant="ghost"
          className="advanced"
          onClick={() => setAdvancedDialogOpen(true)}
        >
          <Settings2
            data-icon="inline-start"
            className="mr-2 h-4 w-4"
          />
          Advanced Options
        </Button>
      </div>
{/* ``` */}


      {/* -----------------------------------------------------------------
          The old "Upload Satellite Image" confirmation Dialog has been
          removed on purpose: attaching is now a single click -> native
          picker -> instant preview chip, with no extra confirmation step.
          The Advanced Options dialog is unchanged.
      ------------------------------------------------------------------ */}

      <Dialog open={isAdvancedDialogOpen} onOpenChange={setAdvancedDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Advanced Analysis Options</DialogTitle>
            <DialogDescription>Configure analysis parameters for more precise results.</DialogDescription>
          </DialogHeader>
          <div className="grid gap-4 py-4">
            <div className="grid gap-2">
              {/* BACKEND: cloudCoverMax param in analysis request */}
              <label className="text-sm font-medium">Cloud Cover Threshold (%)</label>
              <Input
                type="number"
                min="0" max="100"
                value={cloudCover}
                onChange={(e) => setCloudCover(Number(e.target.value))}
              />
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div className="grid gap-2">
                {/* BACKEND: dateFrom, dateTo params in analysis request */}
                <label className="text-sm font-medium">Date From</label>
                <Input type="date" value={dateFrom} onChange={(e) => setDateFrom(e.target.value)} />
              </div>
              <div className="grid gap-2">
                <label className="text-sm font-medium">Date To</label>
                <Input type="date" value={dateTo} onChange={(e) => setDateTo(e.target.value)} />
              </div>
            </div>

            <div className="grid gap-2">
              {/* BACKEND: bands[] param in analysis request */}
              <label className="text-sm font-medium">Band Selection</label>
              <div className="flex flex-wrap gap-2">
                {allBands.map((band) => (
                  <Badge
                    key={band}
                    variant={selectedBands.includes(band) ? "default" : "outline"}
                    className="cursor-pointer"
                    onClick={() => toggleBand(band)}
                  >
                    {selectedBands.includes(band) && <Check className="mr-1 h-3 w-3" />}
                    {band}
                  </Badge>
                ))}
              </div>
            </div>

            <div className="grid gap-2">
              {/* BACKEND: resolutionOverride param in analysis request */}
              <label className="text-sm font-medium">Resolution Override (m/px)</label>
              <Input placeholder="e.g. 10" value={resolution} onChange={(e) => setResolution(e.target.value)} />
            </div>
          </div>
          <DialogFooter>
            <DialogClose ><Button variant="outline">Cancel</Button></DialogClose>
            <Button onClick={() => setAdvancedDialogOpen(false)}>Apply Settings</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* -----------------------------------------------------------------
          Local styles for the new attachment chips + shimmering send
          button. Scoped with styled-jsx so nothing outside this file /
          no shared component is touched.
      ------------------------------------------------------------------ */}
      <style jsx>{`
        .attachment-strip {
          display: flex;
          gap: 8px;
          overflow-x: auto;
          padding: 4px 4px 10px;
          margin-bottom: 4px;
        }

        .attachment-chip {
          position: relative;
          flex: 0 0 auto;
          width: 64px;
          height: 64px;
          border-radius: 10px;
          overflow: hidden;
          border: 1px solid rgba(0, 0, 0, 0.08);
          background: rgba(0, 0, 0, 0.03);
        }

        .attachment-chip-img {
          width: 100%;
          height: 100%;
          object-fit: cover;
          display: block;
        }

        .attachment-chip-name {
          position: absolute;
          inset: auto 0 0 0;
          font-size: 9px;
          line-height: 1.2;
          padding: 2px 4px;
          background: linear-gradient(to top, rgba(0, 0, 0, 0.55), transparent);
          color: #fff;
          white-space: nowrap;
          overflow: hidden;
          text-overflow: ellipsis;
        }

        .attachment-chip-remove {
          position: absolute;
          top: 2px;
          right: 2px;
          width: 16px;
          height: 16px;
          border-radius: 9999px;
          display: flex;
          align-items: center;
          justify-content: center;
          background: rgba(0, 0, 0, 0.6);
          color: #fff;
          border: none;
          cursor: pointer;
          transition: transform 120ms ease, background 120ms ease;
        }

        .attachment-chip-remove:hover {
          background: rgba(0, 0, 0, 0.85);
          transform: scale(1.08);
        }

        @keyframes attachmentIn {
          from {
            opacity: 0;
            transform: scale(0.85) translateY(4px);
          }
          to {
            opacity: 1;
            transform: scale(1) translateY(0);
          }
        }

        /* Send button: subtle press feedback + shimmering "generating" state */
        :global(.send-button) {
          transition: transform 120ms ease, box-shadow 120ms ease;
        }

        :global(.send-button:not(:disabled):active) {
          transform: scale(0.92);
        }

        :global(.send-button-icon) {
          transition: transform 150ms ease;
        }

        :global(.send-button:not(:disabled):hover .send-button-icon) {
          transform: translateY(-1px);
        }

        :global(.send-button-generating) {
          position: relative;
          overflow: hidden;
          background: linear-gradient(
            110deg,
            var(--primary, #111) 0%,
            var(--primary, #111) 40%,
            rgba(255, 255, 255, 0.55) 50%,
            var(--primary, #111) 60%,
            var(--primary, #111) 100%
          );
          background-size: 220% 100%;
          animation: shimmerSweep 1.6s ease-in-out infinite;
        }

        @keyframes shimmerSweep {
          0% {
            background-position: 200% 0;
          }
          100% {
            background-position: -20% 0;
          }
        }

        @media (prefers-reduced-motion: reduce) {
          .attachment-chip {
            animation: none !important;
          }
          :global(.send-button-generating) {
            animation: none !important;
          }
        }
      `}</style>
    </>
  )
}
