import * as Tone from 'tone'

export type PlaybackNote = {
  midi: number
  start: number
  duration: number
}

export type PlaybackPayload = {
  pattern_id: string
  instrument: string
  bpm: number
  duration_sec: number
  notes: PlaybackNote[]
}

let sampler: Tone.Sampler | null = null
let loading: Promise<Tone.Sampler> | null = null
let activeId: string | null = null

function midiToNoteName(midi: number): string {
  return Tone.Frequency(midi, 'midi').toNote()
}

async function getPiano(): Promise<Tone.Sampler> {
  if (sampler) return sampler
  if (loading) return loading

  loading = new Promise<Tone.Sampler>((resolve, reject) => {
    const s = new Tone.Sampler({
      urls: {
        A0: 'A0.mp3',
        C1: 'C1.mp3',
        'D#1': 'Ds1.mp3',
        'F#1': 'Fs1.mp3',
        A1: 'A1.mp3',
        C2: 'C2.mp3',
        'D#2': 'Ds2.mp3',
        'F#2': 'Fs2.mp3',
        A2: 'A2.mp3',
        C3: 'C3.mp3',
        'D#3': 'Ds3.mp3',
        'F#3': 'Fs3.mp3',
        A3: 'A3.mp3',
        C4: 'C4.mp3',
        'D#4': 'Ds4.mp3',
        'F#4': 'Fs4.mp3',
        A4: 'A4.mp3',
        C5: 'C5.mp3',
        'D#5': 'Ds5.mp3',
        'F#5': 'Fs5.mp3',
        A5: 'A5.mp3',
        C6: 'C6.mp3',
        'D#6': 'Ds6.mp3',
        'F#6': 'Fs6.mp3',
        A6: 'A6.mp3',
        C7: 'C7.mp3',
        'D#7': 'Ds7.mp3',
        'F#7': 'Fs7.mp3',
        A7: 'A7.mp3',
        C8: 'C8.mp3',
      },
      baseUrl: 'https://tonejs.github.io/audio/salamander/',
      onload: () => {
        sampler = s
        resolve(s)
      },
      onerror: (err) => reject(err),
    }).toDestination()
    s.volume.value = -6
  })

  return loading
}

export function stopPlayback() {
  Tone.getTransport().stop()
  Tone.getTransport().cancel(0)
  activeId = null
}

export function isPlaying(id?: string): boolean {
  if (!id) return activeId != null && Tone.getTransport().state === 'started'
  return activeId === id && Tone.getTransport().state === 'started'
}

export async function playPattern(
  payload: PlaybackPayload,
  onEnded?: () => void,
): Promise<string> {
  const id = `${payload.pattern_id}__${payload.instrument}`
  await Tone.start()
  const piano = await getPiano()

  stopPlayback()
  activeId = id

  const now = Tone.now() + 0.05
  for (const note of payload.notes) {
    // Keep MIDI in a playable piano range for very low bass notes.
    let midi = note.midi
    while (midi < 21) midi += 12
    while (midi > 108) midi -= 12
    const name = midiToNoteName(midi)
    const dur = Math.max(note.duration, 0.05)
    piano.triggerAttackRelease(name, dur, now + note.start, 0.7)
  }

  const endAt = (payload.duration_sec || 0) + 0.35
  window.setTimeout(() => {
    if (activeId === id) {
      activeId = null
      onEnded?.()
    }
  }, endAt * 1000)

  return id
}

export async function fetchAndPlay(
  patternId: string,
  instrument: string,
  onEnded?: () => void,
): Promise<string> {
  const params = new URLSearchParams({ instrument })
  const res = await fetch(`/api/patterns/${patternId}/playback?${params}`)
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || `Playback fetch failed (${res.status})`)
  }
  const payload = (await res.json()) as PlaybackPayload
  if (!payload.notes?.length) throw new Error('No playable notes in this pattern')
  return playPattern(payload, onEnded)
}
