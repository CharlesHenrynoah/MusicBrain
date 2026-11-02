export type ScenePoint = {
  row: number
  pattern_id: string
  instrument: string
  bpm: number | null
  original_key: string | null
  x: number
  y: number
  z: number
}

export type ScenePayload = {
  method: string
  dims_source: number
  explained_variance_ratio: number[]
  points: ScenePoint[]
  instruments: string[]
}

export type SearchHit = {
  rank: number
  row: number
  pattern_id: string
  instrument: string
  score: number
  metric: string
  higher_is_better?: boolean
}

export type HighlightCoord = {
  row: number
  x: number
  y: number
  z: number
  is_query?: boolean
  is_hit?: boolean
}

export type ChatResponse = {
  message: string
  hits: SearchHit[]
  query_pattern_id: string | null
  query_instrument: string | null
  metric: string | null
  tools_used: string[]
  highlight_coords: HighlightCoord[]
}

export type ChatTurn = {
  role: 'user' | 'assistant'
  content: string
  hits?: SearchHit[]
  tools_used?: string[]
  metric?: string | null
}
