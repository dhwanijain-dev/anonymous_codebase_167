'use client'

import { useSatQuery } from '@/components/satquery-context'
import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'
import { Separator } from '@/components/ui/separator'
import { Avatar, AvatarFallback } from '@/components/ui/avatar'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from '@/components/ui/tooltip'
import {
  ChevronDown,
  ChevronRight,
  ArrowRight,
  Sun,
  Image as ImageIcon,
  Scan,
  Box,
  BarChart3,
  Grid2X2,
  LogOut,
  User,
  Settings,
} from 'lucide-react'

/**
 * BACKEND INTEGRATION: SatelliteThumb Placeholder
 *
 * Replace with actual satellite imagery thumbnails from backend.
 * API Endpoint: GET /api/images/:id/thumbnail
 */
function SatelliteThumb({ className = '' }: { className?: string }) {
  return <div className={`satellite-thumb ${className}`} aria-hidden="true" />
}

/**
 * BACKEND INTEGRATION: Analysis tools
 *
 * API Endpoint: GET /api/tools
 * Response: { tools: { id, name, icon, description }[] }
 *
 * Replace this static array with data from your API.
 */
const tools = [
  ['Image Understanding', ImageIcon],
  ['Change Detection', Scan],
  ['Object Detection', Box],
  ['Time Series Analysis', BarChart3],
  ['Visual Comparison', Grid2X2],
] as const

/**
 * BACKEND INTEGRATION: Example queries
 *
 * API Endpoint: GET /api/examples
 * Response: { examples: { id, text, thumbnailUrl }[] }
 *
 * Replace this static array with personalized examples from your API.
 */
const examples = [
  'Identify water bodies in this image',
  'Show areas with high vegetation density',
  'Detect possible fire incidents',
  'Map buildings in this region',
]

/**
 * RightSidebar — Contains user profile dropdown, active dataset card,
 * analysis tools list, and example queries.
 *
 * All interactive elements are wired to SatQueryContext for state management.
 */
export function RightSidebar() {
  const {
    dataset,
    setDataset,
    availableDatasets,
    selectedTool,
    setSelectedTool,
    setQuery,
    user,
    toggleDarkMode,
  } = useSatQuery()

  return (
    <aside className="right-sidebar">
      {/* Profile bar with user dropdown */}
      {/* BACKEND: User profile from GET /api/auth/me */}
      <div className="profile-bar">
        <Tooltip>
          <TooltipTrigger
            onClick={toggleDarkMode}
            className="inline-flex h-9 w-9 items-center justify-center rounded-md"
          >
            <Sun />
          </TooltipTrigger>
          <TooltipContent>Toggle theme</TooltipContent>
        </Tooltip>
        <Avatar>
          <AvatarFallback>{user.initials}</AvatarFallback>
        </Avatar>
        <span>{user.name}</span>
        {/* Profile dropdown menu */}
        <DropdownMenu>
          <DropdownMenuTrigger >
              <ChevronDown />
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end">
            {/* BACKEND: Navigate to user profile page */}
            <DropdownMenuItem>
              <User /> Profile
            </DropdownMenuItem>
            {/* BACKEND: Navigate to user settings page */}
            <DropdownMenuItem>
              <Settings /> Settings
            </DropdownMenuItem>
            <DropdownMenuSeparator />
            {/* BACKEND: Logout via POST /api/auth/logout */}
            <DropdownMenuItem>
              <LogOut /> Logout
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>

      <div className="right-content">
        {/* Active dataset section */}
        <div className="section-label">ACTIVE DATA</div>
        {/* BACKEND: Active dataset from GET /api/datasets/:id */}
        <Card className="data-card">
          {/* BACKEND: Replace with actual dataset thumbnail */}
          <SatelliteThumb />
          <div>
            <strong>{dataset.name}</strong>
            <span>{dataset.resolution}</span>
          </div>
          {/* BACKEND: Available datasets from GET /api/datasets */}
          <DropdownMenu>
            <DropdownMenuTrigger >
              Change
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              {availableDatasets.map((ds) => (
                <DropdownMenuItem
                  key={ds.id}
                  onClick={() => {
                    /* BACKEND: Update active dataset via PUT /api/user/settings { datasetId } */
                    setDataset(ds)
                  }}
                >
                  <span style={{ fontWeight: ds.id === dataset.id ? 600 : 400 }}>
                    {ds.name}
                  </span>
                  <span style={{ marginLeft: 8, fontSize: 12, color: '#777' }}>
                    {ds.resolution}
                  </span>
                </DropdownMenuItem>
              ))}
            </DropdownMenuContent>
          </DropdownMenu>
        </Card>

        <Separator />

        {/* Analysis tools section */}
        <div className="section-label">ANALYSIS TOOLS</div>
        {/* BACKEND: Available tools from GET /api/tools */}
        <div className="tool-list">
          {tools.map(([name, Icon]) => (
            <button
              key={name}
              onClick={() => {
                /* BACKEND: Activate tool via POST /api/tools/:id/activate */
                setSelectedTool(selectedTool === name ? null : (name as string))
              }}
              className={selectedTool === name ? 'tool-active' : ''}
            >
              <span className="tool-icon"><Icon /></span>
              {name}
              <ChevronRight />
            </button>
          ))}
        </div>

        <Separator />

        {/* Example queries section */}
        <div className="examples-heading">
          {/* BACKEND: Example queries from GET /api/examples */}
          <div className="section-label">EXAMPLE QUERIES</div>
          {/* BACKEND: Paginate examples GET /api/examples?page=2 */}
          <button>See more <ArrowRight /></button>
        </div>
        <div className="example-list">
          {examples.map((text) => (
            <button
              key={text}
              onClick={() => {
                /* BACKEND: Pre-fill query from example, optionally log usage via POST /api/examples/:id/use */
                setQuery(text)
              }}
            >
              {/* BACKEND: Replace with actual example thumbnail from API */}
              <SatelliteThumb />
              <span>{text}</span>
            </button>
          ))}
        </div>
      </div>

      <div className="powered">POWERED BY<br />GEOSPATIAL AI <span /></div>
    </aside>
  )
}
