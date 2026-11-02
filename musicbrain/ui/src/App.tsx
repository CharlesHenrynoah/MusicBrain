import { useEffect, useMemo, useState } from 'react'
import type { FormEvent } from 'react'
import { PatternScene } from './PatternScene'
import { fetchAndPlay, stopPlayback } from './pianoPlayer'
import type {
  ChatResponse,
  ChatTurn,
  HighlightCoord,
  ScenePayload,
  ScenePoint,
  SearchHit,
} from './types'
import './App.css'

const SUGGESTIONS = [
  'Introduce the MusicBrain corpus',
  'Find cosine neighbors of 297249',
  'Compare 297249 and 297250 musically',
  'Synth lead patterns between 130 and 140 BPM',
  'Hybrid search around 281581',
]

const API_KEY_STORAGE = 'musicbrain_inception_api_key'

export default function App() {
  const [scene, setScene] = useState<ScenePayload | null>(null)
  const [apiOk, setApiOk] = useState<boolean | null>(null)
  const [apiKey, setApiKey] = useState(
    () => localStorage.getItem(API_KEY_STORAGE) ?? '',
  )
  const [showKey, setShowKey] = useState(false)
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [turns, setTurns] = useState<ChatTurn[]>([
    {
      role: 'assistant',
      content:
        "Hi — I'm the MusicBrain agent (Mercury 3). Paste your Inception API key above, then ask for a search; the 3D scene will highlight the results.",
    },
  ])
  const [highlights, setHighlights] = useState<HighlightCoord[]>([])
  const [selectedRow, setSelectedRow] = useState<number | null>(null)
  const [lastMetric, setLastMetric] = useState<string | null>(null)
  const [playingId, setPlayingId] = useState<string | null>(null)
  const [playError, setPlayError] = useState<string | null>(null)

  const selectedPoint = useMemo(
    () => scene?.points.find((p) => p.row === selectedRow) ?? null,
    [scene, selectedRow],
  )

  async function togglePlay(patternId: string, instrument: string) {
    const id = `${patternId}__${instrument}`
    setPlayError(null)
    if (playingId === id) {
      stopPlayback()
      setPlayingId(null)
      return
    }
    try {
      setPlayingId(id)
      await fetchAndPlay(patternId, instrument, () => setPlayingId(null))
    } catch (err) {
      setPlayingId(null)
      setPlayError(err instanceof Error ? err.message : String(err))
    }
  }

  function PlayButton({ hit }: { hit: Pick<SearchHit, 'pattern_id' | 'instrument'> }) {
    const id = `${hit.pattern_id}__${hit.instrument}`
    const active = playingId === id
    return (
      <button
        type="button"
        className={`play-btn ${active ? 'active' : ''}`}
        title={active ? 'Stop piano' : 'Play on piano'}
        onClick={(e) => {
          e.stopPropagation()
          void togglePlay(hit.pattern_id, hit.instrument)
        }}
      >
        {active ? '■' : '▶'}
      </button>
    )
  }

  function persistApiKey(value: string) {
    setApiKey(value)
    if (value.trim()) localStorage.setItem(API_KEY_STORAGE, value.trim())
    else localStorage.removeItem(API_KEY_STORAGE)
  }

  useEffect(() => {
    fetch('/api/health')
      .then((r) => r.json())
      .then(() => setApiOk(true))
      .catch(() => setApiOk(false))

    fetch('/api/corpus/3d')
      .then((r) => {
        if (!r.ok) throw new Error('corpus 3d')
        return r.json()
      })
      .then((data: ScenePayload) => setScene(data))
      .catch(() => setError('Could not load the 3D corpus projection.'))
  }, [])

  const variance = useMemo(() => {
    if (!scene) return null
    const total = scene.explained_variance_ratio.reduce((a, b) => a + b, 0)
    return Math.round(total * 100)
  }, [scene])

  async function sendPrompt(text: string) {
    const prompt = text.trim()
    if (!prompt || loading) return
    if (!apiKey.trim()) {
      setError('Paste your Inception (Mercury 3) API key before sending a prompt.')
      return
    }
    setLoading(true)
    setError(null)
    setInput('')
    const history = turns.map(({ role, content }) => ({ role, content }))
    setTurns((prev) => [...prev, { role: 'user', content: prompt }])

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: prompt,
          history,
          api_key: apiKey.trim(),
        }),
      })
      const data = (await res.json()) as ChatResponse & { detail?: string }
      if (!res.ok) {
        const detail =
          typeof data.detail === 'string'
            ? data.detail
            : Array.isArray(data.detail)
              ? JSON.stringify(data.detail)
              : 'Agent error'
        throw new Error(detail)
      }

      setTurns((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: data.message,
          hits: data.hits,
          tools_used: data.tools_used,
          metric: data.metric,
        },
      ])
      setHighlights(data.highlight_coords || [])
      setLastMetric(data.metric)
      const queryHit = data.highlight_coords?.find((h) => h.is_query)
      if (queryHit) setSelectedRow(queryHit.row)
      else if (data.hits?.[0]) setSelectedRow(data.hits[0].row)
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err)
      setError(msg)
      setTurns((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: `Error: ${msg}. Check that the API is running and your Inception key is valid.`,
        },
      ])
    } finally {
      setLoading(false)
    }
  }

  function onSubmit(e: FormEvent) {
    e.preventDefault()
    void sendPrompt(input)
  }

  function onSelectPoint(point: ScenePoint) {
    setSelectedRow(point.row)
  }

  return (
    <div className="app">
      <aside className="panel">
        <header className="brand">
          <div className="tag mono">Inception · Mercury 3</div>
          <h1>MusicBrain</h1>
          <p>LangChain agent connected to the 32D search engine + musical events.</p>
          <div className="key-box">
            <label className="mono" htmlFor="inception-key">
              Inception API key
            </label>
            <div className="key-row">
              <input
                id="inception-key"
                type={showKey ? 'text' : 'password'}
                value={apiKey}
                onChange={(e) => persistApiKey(e.target.value)}
                placeholder="paste your key here…"
                autoComplete="off"
                spellCheck={false}
              />
              <button type="button" className="ghost" onClick={() => setShowKey((v) => !v)}>
                {showKey ? 'Hide' : 'Show'}
              </button>
            </div>
            <div className={`key-status mono ${apiKey.trim() ? 'ok' : ''}`}>
              {apiKey.trim() ? 'key saved locally' : 'key required to chat'}
            </div>
          </div>
        </header>

        <div className="suggestions">
          {SUGGESTIONS.map((s) => (
            <button
              key={s}
              type="button"
              onClick={() => void sendPrompt(s)}
              disabled={loading || !apiKey.trim()}
            >
              {s}
            </button>
          ))}
        </div>

        <div className="chat">
          {turns.map((turn, i) => (
            <div key={i} className={`bubble ${turn.role}`}>
              {turn.content}
              {turn.tools_used && turn.tools_used.length > 0 && (
                <div className="meta mono">tools: {turn.tools_used.join(' · ')}</div>
              )}
              {turn.hits && turn.hits.length > 0 && (
                <ul className="hits mono">
                  {turn.hits.slice(0, 8).map((h) => (
                    <li key={`${h.row}-${h.rank}`}>
                      <span className="hit-main">
                        <PlayButton hit={h} />
                        <button
                          type="button"
                          className="hit-select"
                          onClick={() => setSelectedRow(h.row)}
                        >
                          #{h.rank} {h.pattern_id} · {h.instrument}
                        </button>
                      </span>
                      <span>{h.score.toFixed(3)}</span>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          ))}
          {loading && <div className="bubble">Searching…</div>}
          {playError && <div className="bubble">Piano: {playError}</div>}
        </div>

        <form className="composer" onSubmit={onSubmit}>
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="e.g. cosine neighbors of 297249, or Am patterns around 140 BPM"
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault()
                void sendPrompt(input)
              }
            }}
          />
          <button type="submit" disabled={loading || !input.trim() || !apiKey.trim()}>
            Send
          </button>
        </form>
      </aside>

      <section className="stage">
        <div className="stage-header">
          <div className="stage-card">
            <h2>32D → 3D space (PCA)</h2>
            <p>
              {scene ? `${scene.points.length} patterns` : 'Loading…'}
              {variance != null ? ` · explained variance ~${variance}%` : ''}
              {lastMetric ? ` · ${lastMetric}` : ''}
            </p>
          </div>
          <div className="stage-card stage-actions">
            <div
              className={`status-pill mono ${apiOk === true ? 'ok' : apiOk === false ? 'err' : ''}`}
            >
              API {apiOk === true ? 'online' : apiOk === false ? 'offline' : '…'}
            </div>
            {selectedPoint && (
              <button
                type="button"
                className={`play-btn large ${
                  playingId === `${selectedPoint.pattern_id}__${selectedPoint.instrument}`
                    ? 'active'
                    : ''
                }`}
                onClick={() =>
                  void togglePlay(selectedPoint.pattern_id, selectedPoint.instrument)
                }
              >
                {playingId === `${selectedPoint.pattern_id}__${selectedPoint.instrument}`
                  ? '■ Stop piano'
                  : `▶ Play #${selectedPoint.pattern_id}`}
              </button>
            )}
            {error && <p>{error}</p>}
            {playError && <p>{playError}</p>}
          </div>
        </div>

        {scene && (
          <PatternScene
            points={scene.points}
            highlights={highlights}
            selectedRow={selectedRow}
            onSelect={onSelectPoint}
          />
        )}
      </section>
    </div>
  )
}
