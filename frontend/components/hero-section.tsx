'use client'


import { useEffect, useRef } from 'react'
import { useSatQuery } from '@/components/satquery-context'
import { Card } from '@/components/ui/card'
import { Trees, Waves, Leaf, Mountain, ArrowRight, Loader2 } from 'lucide-react'
import dynamic from "next/dynamic";

const World = dynamic(
  () => import("@/components/ui/globe").then((mod) => mod.World),
  {
    ssr: false,
    loading: () => null,
  }
);
/**
 * BACKEND INTEGRATION: SatelliteThumb Placeholder
 *
 * Replace this CSS-only placeholder with actual satellite imagery
 * loaded from your backend or a tile server.
 *
 * API Endpoint: GET /api/tiles/:z/:x/:y
 * Or: GET /api/images/:imageId/thumbnail
 */
function SatelliteThumb({ className = '' }: { className?: string }) {
  return <div className={`satellite-thumb ${className}`} aria-hidden="true" />
}
const colors = ["#919494", "#64666a", "#808087"];
  const sampleArcs = [
    {
      order: 1,
      startLat: -19.885592,
      startLng: -43.951191,
      endLat: -22.9068,
      endLng: -43.1729,
      arcAlt: 0.1,
      color: colors[Math.floor(Math.random() * (colors.length - 1))],
    },
    {
      order: 1,
      startLat: 28.6139,
      startLng: 77.209,
      endLat: 3.139,
      endLng: 101.6869,
      arcAlt: 0.2,
      color: colors[Math.floor(Math.random() * (colors.length - 1))],
    },
    {
      order: 1,
      startLat: -19.885592,
      startLng: -43.951191,
      endLat: -1.303396,
      endLng: 36.852443,
      arcAlt: 0.5,
      color: colors[Math.floor(Math.random() * (colors.length - 1))],
    },
    {
      order: 2,
      startLat: 1.3521,
      startLng: 103.8198,
      endLat: 35.6762,
      endLng: 139.6503,
      arcAlt: 0.2,
      color: colors[Math.floor(Math.random() * (colors.length - 1))],
    },
    {
      order: 2,
      startLat: 51.5072,
      startLng: -0.1276,
      endLat: 3.139,
      endLng: 101.6869,
      arcAlt: 0.3,
      color: colors[Math.floor(Math.random() * (colors.length - 1))],
    },
    {
      order: 2,
      startLat: -15.785493,
      startLng: -47.909029,
      endLat: 36.162809,
      endLng: -115.119411,
      arcAlt: 0.3,
      color: colors[Math.floor(Math.random() * (colors.length - 1))],
    },
    {
      order: 3,
      startLat: -33.8688,
      startLng: 151.2093,
      endLat: 22.3193,
      endLng: 114.1694,
      arcAlt: 0.3,
      color: colors[Math.floor(Math.random() * (colors.length - 1))],
    },
    {
      order: 3,
      startLat: 21.3099,
      startLng: -157.8581,
      endLat: 40.7128,
      endLng: -74.006,
      arcAlt: 0.3,
      color: colors[Math.floor(Math.random() * (colors.length - 1))],
    },
    {
      order: 3,
      startLat: -6.2088,
      startLng: 106.8456,
      endLat: 51.5072,
      endLng: -0.1276,
      arcAlt: 0.3,
      color: colors[Math.floor(Math.random() * (colors.length - 1))],
    },
    {
      order: 4,
      startLat: 11.986597,
      startLng: 8.571831,
      endLat: -15.595412,
      endLng: -56.05918,
      arcAlt: 0.5,
      color: colors[Math.floor(Math.random() * (colors.length - 1))],
    },
    {
      order: 4,
      startLat: -34.6037,
      startLng: -58.3816,
      endLat: 22.3193,
      endLng: 114.1694,
      arcAlt: 0.7,
      color: colors[Math.floor(Math.random() * (colors.length - 1))],
    },
    {
      order: 4,
      startLat: 51.5072,
      startLng: -0.1276,
      endLat: 48.8566,
      endLng: -2.3522,
      arcAlt: 0.1,
      color: colors[Math.floor(Math.random() * (colors.length - 1))],
    },
    {
      order: 5,
      startLat: 14.5995,
      startLng: 120.9842,
      endLat: 51.5072,
      endLng: -0.1276,
      arcAlt: 0.3,
      color: colors[Math.floor(Math.random() * (colors.length - 1))],
    },
    {
      order: 5,
      startLat: 1.3521,
      startLng: 103.8198,
      endLat: -33.8688,
      endLng: 151.2093,
      arcAlt: 0.2,
      color: colors[Math.floor(Math.random() * (colors.length - 1))],
    },
    {
      order: 5,
      startLat: 34.0522,
      startLng: -118.2437,
      endLat: 48.8566,
      endLng: -2.3522,
      arcAlt: 0.2,
      color: colors[Math.floor(Math.random() * (colors.length - 1))],
    },
    {
      order: 6,
      startLat: -15.432563,
      startLng: 28.315853,
      endLat: 1.094136,
      endLng: -63.34546,
      arcAlt: 0.7,
      color: colors[Math.floor(Math.random() * (colors.length - 1))],
    },
    {
      order: 6,
      startLat: 37.5665,
      startLng: 126.978,
      endLat: 35.6762,
      endLng: 139.6503,
      arcAlt: 0.1,
      color: colors[Math.floor(Math.random() * (colors.length - 1))],
    },
    {
      order: 6,
      startLat: 22.3193,
      startLng: 114.1694,
      endLat: 51.5072,
      endLng: -0.1276,
      arcAlt: 0.3,
      color: colors[Math.floor(Math.random() * (colors.length - 1))],
    },
    {
      order: 7,
      startLat: -19.885592,
      startLng: -43.951191,
      endLat: -15.595412,
      endLng: -56.05918,
      arcAlt: 0.1,
      color: colors[Math.floor(Math.random() * (colors.length - 1))],
    },
    {
      order: 7,
      startLat: 48.8566,
      startLng: -2.3522,
      endLat: 52.52,
      endLng: 13.405,
      arcAlt: 0.1,
      color: colors[Math.floor(Math.random() * (colors.length - 1))],
    },
    {
      order: 7,
      startLat: 52.52,
      startLng: 13.405,
      endLat: 34.0522,
      endLng: -118.2437,
      arcAlt: 0.2,
      color: colors[Math.floor(Math.random() * (colors.length - 1))],
    },
    {
      order: 8,
      startLat: -8.833221,
      startLng: 13.264837,
      endLat: -33.936138,
      endLng: 18.436529,
      arcAlt: 0.2,
      color: colors[Math.floor(Math.random() * (colors.length - 1))],
    },
    {
      order: 8,
      startLat: 49.2827,
      startLng: -123.1207,
      endLat: 52.3676,
      endLng: 4.9041,
      arcAlt: 0.2,
      color: colors[Math.floor(Math.random() * (colors.length - 1))],
    },
    {
      order: 8,
      startLat: 1.3521,
      startLng: 103.8198,
      endLat: 40.7128,
      endLng: -74.006,
      arcAlt: 0.5,
      color: colors[Math.floor(Math.random() * (colors.length - 1))],
    },
    {
      order: 9,
      startLat: 51.5072,
      startLng: -0.1276,
      endLat: 34.0522,
      endLng: -118.2437,
      arcAlt: 0.2,
      color: colors[Math.floor(Math.random() * (colors.length - 1))],
    },
    {
      order: 9,
      startLat: 22.3193,
      startLng: 114.1694,
      endLat: -22.9068,
      endLng: -43.1729,
      arcAlt: 0.7,
      color: colors[Math.floor(Math.random() * (colors.length - 1))],
    },
    {
      order: 9,
      startLat: 1.3521,
      startLng: 103.8198,
      endLat: -34.6037,
      endLng: -58.3816,
      arcAlt: 0.5,
      color: colors[Math.floor(Math.random() * (colors.length - 1))],
    },
    {
      order: 10,
      startLat: -22.9068,
      startLng: -43.1729,
      endLat: 28.6139,
      endLng: 77.209,
      arcAlt: 0.7,
      color: colors[Math.floor(Math.random() * (colors.length - 1))],
    },
    {
      order: 10,
      startLat: 34.0522,
      startLng: -118.2437,
      endLat: 31.2304,
      endLng: 121.4737,
      arcAlt: 0.3,
      color: colors[Math.floor(Math.random() * (colors.length - 1))],
    },
    {
      order: 10,
      startLat: -6.2088,
      startLng: 106.8456,
      endLat: 52.3676,
      endLng: 4.9041,
      arcAlt: 0.3,
      color: colors[Math.floor(Math.random() * (colors.length - 1))],
    },
    {
      order: 11,
      startLat: 41.9028,
      startLng: 12.4964,
      endLat: 34.0522,
      endLng: -118.2437,
      arcAlt: 0.2,
      color: colors[Math.floor(Math.random() * (colors.length - 1))],
    },
    {
      order: 11,
      startLat: -6.2088,
      startLng: 106.8456,
      endLat: 31.2304,
      endLng: 121.4737,
      arcAlt: 0.2,
      color: colors[Math.floor(Math.random() * (colors.length - 1))],
    },
    {
      order: 11,
      startLat: 22.3193,
      startLng: 114.1694,
      endLat: 1.3521,
      endLng: 103.8198,
      arcAlt: 0.2,
      color: colors[Math.floor(Math.random() * (colors.length - 1))],
    },
    {
      order: 12,
      startLat: 34.0522,
      startLng: -118.2437,
      endLat: 37.7749,
      endLng: -122.4194,
      arcAlt: 0.1,
      color: colors[Math.floor(Math.random() * (colors.length - 1))],
    },
    {
      order: 12,
      startLat: 35.6762,
      startLng: 139.6503,
      endLat: 22.3193,
      endLng: 114.1694,
      arcAlt: 0.2,
      color: colors[Math.floor(Math.random() * (colors.length - 1))],
    },
    {
      order: 12,
      startLat: 22.3193,
      startLng: 114.1694,
      endLat: 34.0522,
      endLng: -118.2437,
      arcAlt: 0.3,
      color: colors[Math.floor(Math.random() * (colors.length - 1))],
    },
    {
      order: 13,
      startLat: 52.52,
      startLng: 13.405,
      endLat: 22.3193,
      endLng: 114.1694,
      arcAlt: 0.3,
      color: colors[Math.floor(Math.random() * (colors.length - 1))],
    },
    {
      order: 13,
      startLat: 11.986597,
      startLng: 8.571831,
      endLat: 35.6762,
      endLng: 139.6503,
      arcAlt: 0.3,
      color: colors[Math.floor(Math.random() * (colors.length - 1))],
    },
    {
      order: 13,
      startLat: -22.9068,
      startLng: -43.1729,
      endLat: -34.6037,
      endLng: -58.3816,
      arcAlt: 0.1,
      color: colors[Math.floor(Math.random() * (colors.length - 1))],
    },
    {
      order: 14,
      startLat: -33.936138,
      startLng: 18.436529,
      endLat: 21.395643,
      endLng: 39.883798,
      arcAlt: 0.3,
      color: colors[Math.floor(Math.random() * (colors.length - 1))],
    },
  ];
 
  const globeConfig = {
  // Smaller points = less distracting background
    pointSize: 2,

    // Deep space / Earth observation palette
    globeColor: "#1c2124",

    // Subtle atmospheric edge
    // showAtmosphere: false,
    atmosphereColor: "#08665c",
    atmosphereAltitude: 0.06,

    // Very subtle emissive surface
    emissive: "#07131A",
    emissiveIntensity: 0.15,

    // Keep the surface relatively matte
    shininess: 0.35,

    // Satellite/analysis grid
    polygonColor: "rgba(91, 224, 214, 0.28)",

    // Lighting
    ambientLight: "#16343B",
    directionalLeftLight: "#9FE7E0",
    directionalTopLight: "#FFFFFF",
    pointLight: "#5EEAD4",

    // Data arcs
    arcTime: 1400,
    arcLength: 0.65,

    // Data activity
    rings: 1,
    maxRings: 5,

    // India / remote-sensing focused starting view
    initialPosition: {
      lat: 22.3193,
      lng: 78.9629,
    },

    // Slow scientific visualization
    autoRotate: true,
    autoRotateSpeed: 0.15,
  };
/**
 * BACKEND INTEGRATION: Category definitions
 *
 * API Endpoint: GET /api/categories
 * Response: { categories: { id, title, description, icon }[] }
 *
 * Replace this static array with data fetched from your API.
 */
const categories = [
  { title: 'Land Use & Land Cover', desc: 'Classify and monitor land use changes', Icon: Trees },
  { title: 'Disaster Monitoring', desc: 'Detect flood, fire, and other disaster impacts', Icon: Waves },
  { title: 'Agriculture & Crops', desc: 'Analyze crop health and vegetation', Icon: Leaf },
  { title: 'Urban & Infrastructure', desc: 'Track urban growth and infrastructure', Icon: Mountain },
]

/**
 * BACKEND INTEGRATION: Query suggestions
 *
 * API Endpoint: GET /api/suggestions
 * Response: { suggestions: string[] }
 *
 * Replace this static array with suggestions from your API.
 * Can be personalized based on user history.
 */
const suggestions = [
  'Detect deforestation in this region',
  'Analyse crop health using NDVI',
  'Compare pre and post flood imagery',
  'Show urban expansion over time',
]

/**
 * HeroSection — Main content area with hero copy, globe art,
 * category cards, query suggestions, and analysis results display.
 *
 * This component reads state from SatQueryContext and does not
 * manage its own state.
 */
export function HeroSection() {
  const { setQuery, sentQuery, isAnalyzing, analysisResults } = useSatQuery()

  // Guards so the "save to recent chats" hook (see effect below) only
  // fires once per completed analysis, not on every re-render.
  const savedResultRef = useRef<unknown>(null)

  useEffect(() => {
    if (!analysisResults || isAnalyzing) return
    if (savedResultRef.current === analysisResults) return
    savedResultRef.current = analysisResults

    /**
     * BACKEND INTEGRATION: Save completed chat to Recent Chats
     * -------------------------------------------------------------
     * This fires exactly once per finished analysis (sentQuery +
     * analysisResults both settled). This is where a finished exchange
     * should be written to the recent-chats list so it's visible next
     * time the user opens the sidebar/history panel.
     *
     * Suggested implementation:
     *   const chatId = currentChatId ?? crypto.randomUUID()
     *   const record = {
     *     id: chatId,
     *     title: sentQuery?.slice(0, 60) ?? 'Untitled analysis',
     *     preview: analysisResults.summary,
     *     datasetId: dataset?.id ?? null,
     *     createdAt: existingRecord?.createdAt ?? Date.now(),
     *     updatedAt: Date.now(),
     *   }
     *   // Remote persistence:
     *   await fetch('/api/chats', {
     *     method: existingRecord ? 'PATCH' : 'POST',
     *     body: JSON.stringify(record),
     *   })
     *   // Local-first fallback:
     *   const recents = JSON.parse(localStorage.getItem('recentChats') ?? '[]')
     *   localStorage.setItem(
     *     'recentChats',
     *     JSON.stringify([record, ...recents.filter((r) => r.id !== chatId)].slice(0, 20))
     *   )
     *   // Then notify whatever renders the sidebar (context/store) so the
     *   // new/updated entry appears without a manual refresh.
     */
  }, [analysisResults, isAnalyzing, sentQuery])

  return (
    <div className="hero-content">
      {/* Hero headline and subtext */}
      <div className="hero-copy">
        <h1>Ask the Earth<br /><em>a deeper question.</em></h1>
        <p>Analyse satellite imagery, maps and geospatial data<br className="desktop-only" /> using natural language.</p>
      </div>

      {/* Globe artwork — CSS placeholder, no external image URL */}
      {/* BACKEND: Replace with actual satellite globe tile from GET /api/tiles/globe */}
      <div className="globe-art" aria-hidden="true"
      // style={{  scale:"2"  , transform:"translateY(100px)"  }}
      >
      
        <World
        globeConfig={globeConfig}
        data={sampleArcs}
      />
      </div>

      {/* Category cards grid */}
      {/* BACKEND: Categories from GET /api/categories */}
      {/* <div className="category-grid">
        {categories.map(({ title, desc, Icon }) => (
          <Card
            className="category-card"
            key={title}
            onClick={() => {
              setQuery(desc)
            }}
            style={{ cursor: 'pointer' }}
          >
            <SatelliteThumb />
            <div className="category-body">
              <div className="category-icon"><Icon /></div>
              <h3>{title}</h3>
              <p>{desc}</p>
            </div>
          </Card>
        ))}
      </div> */}

      {/* Query suggestion buttons */}
      {/* BACKEND: Query suggestions from GET /api/suggestions */}
      {/* <div className="query-suggestions">
        {suggestions.map((item) => (
          <button key={item} onClick={() => setQuery(item)}>
            {item}
            <ArrowRight />
          </button>
        ))}
      </div> */}

      {/* Sent query display — fades/slides in the moment a query is sent */}
      {/* BACKEND: Current analysis status from GET /api/analysis/:id/status */}
      {/* {sentQuery && (
        <div className="sent-query hero-fade-in" role="status">
          Analysing: <strong>{sentQuery}</strong>
        </div>
      )} */}

      {/* Analysis loading state — pulsing dot instead of a static spinner
          label, to read as "working" rather than "stuck" */}
        {/* {isAnalyzing && (
          <div className="analysis-loading hero-fade-in" role="status" aria-live="polite">
            <Loader2 className="animate-spin" />
            <span>Processing satellite data...</span>
          </div>
        )} */}

      {/* Analysis results display — settles in with a short fade + slight
          upward slide once results are ready, answering the loading state
          rather than appearing unannounced. */}
      {/* BACKEND: Analysis results from GET /api/analysis/:id/results */}
      {/* {analysisResults && !isAnalyzing && (
        <div className="analysis-result hero-result-in" role="status" aria-live="polite">
          <strong>Analysis Complete:</strong> {analysisResults.summary}
        </div>
      )} */}

      <style jsx>{`
        .hero-fade-in {
          animation: heroFadeIn 260ms ease-out;
        }
        
        .hero-result-in {
          animation: heroResultIn 320ms cubic-bezier(0.16, 1, 0.3, 1);
        }

        @keyframes heroFadeIn {
          from {
            opacity: 0;
          }
          to {
            opacity: 1;
          }
        }

        @keyframes heroResultIn {
          from {
            opacity: 0;
            transform: translateY(6px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }

        @media (prefers-reduced-motion: reduce) {
          .hero-fade-in,
          .hero-result-in {
            animation: none !important;
          }
        }
      `}</style>
    </div>
  )
}