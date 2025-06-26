import React, { useState, useMemo, useEffect, useRef } from 'react';
import { Stage, Layer, Line, Circle, Image as KonvaImage } from 'react-konva';
import useImage from 'use-image';
import './ContourEditor.css'

interface ContourEditorProps {
  imageUrl: string;
  initialContour: [number, number][];
  onUpdate?: (updatedPoints: [number, number][]) => void;
}

const MAX_WIDTH = 500;

const ContourEditor: React.FC<ContourEditorProps> = ({ imageUrl, initialContour, onUpdate }) => {
  const [image] = useImage(imageUrl);
  const [points, setPoints] = useState<[number, number][]>(initialContour);
  const [repelMode, setRepelMode] = useState(false);
  const [isRepelling, setIsRepelling] = useState(false);
  const [cursorPos, setCursorPos] = useState<{ x: number; y: number } | null>(null);
  const [repelStrength, setRepelStrength] = useState(10); // Initialize repel strength
  const [repelRadius, setRepelRadius] = useState(10); // Initialize repel radius
  const [undoStack, setUndoStack] = useState<[number, number][][]>([]);

  const animationFrameId = useRef<number | null>(null);

  const scale = useMemo(() => {
    if (!image) return 1;
    return image.width > MAX_WIDTH ? MAX_WIDTH / image.width : 1;
  }, [image]);

  const scaledPoints = useMemo(() => points.map(([x, y]) => [x * scale, y * scale]), [points, scale]);
  const flattenedPoints = useMemo(() => scaledPoints.flat(), [scaledPoints]);

  const pushToUndoStack = (current: [number, number][]) => {
    setUndoStack((prev) => [...prev, current.map(p => [...p] as [number, number])]);
  };
  

  useEffect(() => {
    setPoints(initialContour);
  }, [initialContour]);

  const repelPoints = () => {
    if (!repelMode || !isRepelling || !cursorPos) {
      return;
    }

    // pushToUndoStack(points);

    setPoints(prevPoints =>
      prevPoints.map(([x, y]) => {
        const dx = x * scale - cursorPos.x;
        const dy = y * scale - cursorPos.y;
        const dist = Math.sqrt(dx * dx + dy * dy);

        if (dist < repelRadius) {
          const force = ((repelRadius - dist) / repelRadius) * repelStrength;
          const angle = Math.atan2(dy, dx);
          const newX = x + (force * Math.cos(angle)) / scale;
          const newY = y + (force * Math.sin(angle)) / scale;
          return [newX, newY];
        }
        return [x, y];
      })
    );

    animationFrameId.current = requestAnimationFrame(repelPoints);
  };

  // Trigger repel animation on state change
  useEffect(() => {
    if (isRepelling && repelMode) {
      animationFrameId.current = requestAnimationFrame(repelPoints);
    }
    return () => {
      if (animationFrameId.current) {
        cancelAnimationFrame(animationFrameId.current);
        animationFrameId.current = null;
      }
    };
  }, [isRepelling, cursorPos, repelMode]);

  const handleMouseDown = (e: any) => {
    pushToUndoStack(points);
    if (repelMode) {
      const pos = e.target.getStage().getPointerPosition();
      if (pos) {
        setCursorPos(pos);
        setIsRepelling(true);
      }
    }
  };

  const handleMouseUp = () => {
    // pushToUndoStack(points);
    if (repelMode) {
      setIsRepelling(false);
      setCursorPos(null);
      if (animationFrameId.current) {
        cancelAnimationFrame(animationFrameId.current);
        animationFrameId.current = null;
      }
      if (onUpdate) onUpdate(points);
    }
  };

  const handleMouseMove = (e: any) => {
    if (repelMode && isRepelling) {
      const pos = e.target.getStage().getPointerPosition();
      if (pos) {
        setCursorPos(pos);
      }
    }
  };

  const handleDragMove = (index: number, x: number, y: number) => {
    if (repelMode) return;

    // pushToUndoStack(points);

    const newPoints = [...points];
    newPoints[index] = [x / scale, y / scale];

    setPoints(newPoints);
    if (onUpdate) {
      onUpdate(newPoints);
    //   console.log(stage_width)
    }
  };

//   const stage_width = image ? image.width * scale : MAX_WIDTH;
//   const stage_height = image ? image.height * scale : 200;

  return (
    <div>
      <div className="mb-2 flex gap-2">
        <button
          onClick={() => setRepelMode(!repelMode)}
          className="px-3 py-1 bg-blue-500 text-white rounded"
        >
          {repelMode ? 'Switch to Drag Mode' : 'Switch to Repel Mode'}
        </button>

        <button
        onClick={() => {
            if (undoStack.length > 0) {
            const last = undoStack[undoStack.length - 1];
            setPoints(last);
            setUndoStack(prev => prev.slice(0, -1));
            if (onUpdate) onUpdate(last);
            }
        }}
        disabled={undoStack.length === 0}
        className={`px-3 py-1 ${undoStack.length === 0 ? 'bg-gray-400' : 'bg-red-500'} text-white rounded`}
        >
        Undo
        </button>

        {repelMode && (
          <>
            <label className="ml-4 text-sm">
              Strength:
              <input
                type="range"
                min={10}
                max={20}
                value={repelStrength}
                onChange={(e) => setRepelStrength(parseInt(e.target.value))}
                className="ml-2"
              />
              {repelStrength}
            </label>

            <label className="ml-4 text-sm">
              Radius:
              <input
                type="range"
                min={2}
                max={40}
                value={repelRadius}
                onChange={(e) => setRepelRadius(parseInt(e.target.value))}
                className="ml-2"
              />
              {repelRadius}
            </label>
          </>
        )}
      </div>

      <Stage
        width={image ? image.width * scale : MAX_WIDTH}
        height={image ? image.height * scale : 200}
        onMouseDown={handleMouseDown}
        onMouseUp={handleMouseUp}
        onMouseMove={handleMouseMove}
        className='stage'
        style={{ border: '1px solid #ccc', cursor: repelMode ? 'crosshair' : 'default' }}
      >
        <Layer>
          {image && (
            <KonvaImage
              image={image}
              width={image.width * scale}
              height={image.height * scale}
            //   {stage_width = image ? image.width * scale : MAX_WIDTH}
            //   x = {((stage_width) - image.width*scale) / 2}
            //     x = {200}
            //   y = {(stage_height - image.height*scale) / 2}
            //   x={stage_width/4}
            />
          )}

          <Line
            points={flattenedPoints}
            closed
            stroke="blue"
            strokeWidth={2}
            tension={0.5}
            lineCap="round"
          />

          {!repelMode &&
            scaledPoints.map(([x, y], i) => (
              <Circle
                key={i}
                x={x}
                y={y}
                radius={2}
                fill="red"
                stroke="black"
                strokeWidth={0}
                draggable
                onDragMove={(e) => handleDragMove(i, e.target.x(), e.target.y())}
              />
            ))}

          {/* Visual indicator of influence radius */}
          {repelMode && isRepelling && cursorPos && (
            <Circle
              x={cursorPos.x}
              y={cursorPos.y}
              radius={repelRadius}
              stroke="rgba(0,0,255,0.5)"
              strokeWidth={1}
              dash={[4, 4]}
            />
          )}
        </Layer>
      </Stage>
    </div>
  );
};

export default ContourEditor;
