import { useMemo, useRef, useState } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { OrbitControls, Html } from '@react-three/drei'
import * as THREE from 'three'
import type { HighlightCoord, ScenePoint } from './types'

const INSTRUMENT_GRAY: Record<string, string> = {
  'Synth lead': '#f2f2f2',
  Keys: '#c8c8c8',
  Guitar: '#9a9a9a',
  Bass: '#6e6e6e',
  Bell: '#e8e8e8',
}

type Props = {
  points: ScenePoint[]
  highlights: HighlightCoord[]
  selectedRow: number | null
  onSelect: (point: ScenePoint) => void
}

function Cloud({
  points,
  highlightRows,
  queryRow,
  selectedRow,
  onSelect,
}: {
  points: ScenePoint[]
  highlightRows: Set<number>
  queryRow: number | null
  selectedRow: number | null
  onSelect: (point: ScenePoint) => void
}) {
  const group = useRef<THREE.Group>(null)
  useFrame((_, dt) => {
    if (group.current) group.current.rotation.y += dt * 0.03
  })

  return (
    <group ref={group}>
      {points.map((p) => {
        const isQuery = queryRow === p.row
        const isHit = highlightRows.has(p.row) && !isQuery
        const isSelected = selectedRow === p.row
        const base = INSTRUMENT_GRAY[p.instrument] ?? '#888888'
        const color = isQuery ? '#ffffff' : isHit ? '#ffffff' : base
        const scale = isQuery ? 0.22 : isHit ? 0.16 : isSelected ? 0.14 : 0.065
        return (
          <mesh
            key={`${p.pattern_id}-${p.instrument}`}
            position={[p.x, p.y, p.z]}
            scale={scale}
            onClick={(e) => {
              e.stopPropagation()
              onSelect(p)
            }}
          >
            <sphereGeometry args={[1, 16, 16]} />
            <meshStandardMaterial
              color={color}
              emissive={isQuery || isHit ? '#ffffff' : '#000000'}
              emissiveIntensity={isQuery ? 0.55 : isHit ? 0.35 : 0}
              roughness={0.45}
              metalness={0.05}
              transparent={!isQuery && !isHit}
              opacity={isQuery || isHit || isSelected ? 1 : 0.55}
            />
          </mesh>
        )
      })}
    </group>
  )
}

export function PatternScene({ points, highlights, selectedRow, onSelect }: Props) {
  const [hovered, setHovered] = useState<ScenePoint | null>(null)
  const highlightRows = useMemo(
    () => new Set(highlights.filter((h) => h.is_hit).map((h) => h.row)),
    [highlights],
  )
  const queryRow = useMemo(
    () => highlights.find((h) => h.is_query)?.row ?? null,
    [highlights],
  )

  const selected = points.find((p) => p.row === selectedRow) ?? hovered

  return (
    <div className="scene-wrap">
      <Canvas camera={{ position: [0, 2.5, 16], fov: 50 }}>
        <color attach="background" args={['#000000']} />
        <ambientLight intensity={0.7} />
        <directionalLight position={[6, 8, 4]} intensity={1.0} color="#ffffff" />
        <directionalLight position={[-5, -2, -6]} intensity={0.25} color="#ffffff" />
        <Cloud
          points={points}
          highlightRows={highlightRows}
          queryRow={queryRow}
          selectedRow={selectedRow}
          onSelect={(p) => {
            setHovered(p)
            onSelect(p)
          }}
        />
        <OrbitControls enablePan enableDamping dampingFactor={0.08} />
        {selected && (
          <Html position={[selected.x, selected.y + 0.45, selected.z]} center>
            <div className="point-label mono">
              #{selected.pattern_id} · {selected.instrument}
              {selected.bpm ? ` · ${selected.bpm} BPM` : ''}
              {selected.original_key ? ` · ${selected.original_key}` : ''}
            </div>
          </Html>
        )}
      </Canvas>
      <div className="scene-legend mono">
        <span>
          <i style={{ background: '#f2f2f2' }} /> Synth
        </span>
        <span>
          <i style={{ background: '#c8c8c8' }} /> Keys
        </span>
        <span>
          <i style={{ background: '#ffffff' }} /> Query / hits
        </span>
        <span>
          <i style={{ background: '#6e6e6e' }} /> Bass
        </span>
      </div>
    </div>
  )
}
