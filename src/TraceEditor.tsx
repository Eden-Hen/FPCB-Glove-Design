import React, { useState, useRef, useEffect } from "react";

type Point = [number, number];
type Polygon = Point[];
type TraceData = { [id: string]: Polygon };

interface TraceEditorProps {
  tracedImage: string;
  initialTraces: TraceData;
  finalOutline?: Polygon | null;
  onChange: (traces: TraceData) => void;
  width?: number;
}

const TraceEditor: React.FC<TraceEditorProps> = ({
  tracedImage,
  initialTraces,
  finalOutline,
  onChange,
  width = 500,
}) => {
    const [traces, setTraces] = useState<TraceData>(initialTraces);
    const [dragging, setDragging] = useState<{ regionId: string; pointIdx: number } | null>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const [imgSize, setImgSize] = useState<{ width: number; height: number } | null>(null);

  // Sync internal state with incoming traces
  useEffect(() => {
    setTraces(initialTraces);
  }, [initialTraces]);

  // Load image to get actual size
  useEffect(() => {
    const img = new Image();
    img.onload = () => setImgSize({ width: img.width, height: img.height });
    img.src = tracedImage;
  }, [tracedImage]);

  const handleMouseDown = (regionId: string, pointIdx: number) => {
    setDragging({ regionId, pointIdx });
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!dragging || !svgRef.current || !imgSize) return;

    const svgRect = svgRef.current.getBoundingClientRect();
    const scaleX = imgSize.width / svgRect.width;
    const scaleY = imgSize.height / svgRect.height;
    const x = (e.clientX - svgRect.left) * scaleX;
    const y = (e.clientY - svgRect.top) * scaleY;

    // Create a new copy of the object to avoid direct mutation
    const updatedTraces = { ...traces };
    // Update the specific point in the specific polygon using its key (regionId)
    updatedTraces[dragging.regionId][dragging.pointIdx] = [x, y];

    setTraces(updatedTraces);
    onChange(updatedTraces);
  };

  const handleMouseUp = () => {
    setDragging(null);
  };

  if (!imgSize) return <p>Loading image...</p>;

  return (
    <div style={{ position: "relative", width, margin: "auto" }}>
      <svg
        ref={svgRef}
        width={width}
        height={(imgSize.height / imgSize.width) * width}
        viewBox={`0 0 ${imgSize.width} ${imgSize.height}`}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        style={{
          border: "1px solid #ccc",
          backgroundImage: `url(${tracedImage})`,
          backgroundSize: "contain",
          backgroundRepeat: "no-repeat",
          backgroundPosition: "center",
          display: "block"
        }}
      >
        {Object.keys(traces).map((regionId) => (
          <g key={regionId}>
            <polygon
              points={traces[regionId].map(([x, y]) => `${x},${y}`).join(" ")}
              fill="rgba(0, 0, 255, 0.2)"
              stroke="blue"
              strokeWidth={2}
            />
            {traces[regionId].map(([x, y], ptIdx) => (
              <circle
                key={ptIdx}
                cx={x}
                cy={y}
                r={10}
                fill="white"
                stroke="black"
                strokeWidth={1}
                // Pass the string regionId to the handler
                onMouseDown={() => handleMouseDown(regionId, ptIdx)}
                style={{ cursor: "grab" }}
              />
            ))}
          </g>
        ))}
        {/* If the finalOutline data exists, draw it on top as a new layer */}
        {finalOutline && (
          <polygon
            points={finalOutline.map(([x, y]) => `${x},${y}`).join(" ")}
            fill="none" // Make it hollow so you can see through it
            stroke="rgba(255, 0, 0, 0.7)" // A semi-transparent red is good for outlines
            strokeWidth={10} // A bit thicker to make it clearly visible
            style={{ pointerEvents: "none" }} // This makes the line "un-clickable" so it doesn't interfere with editing the traces underneath
          />
        )}
      </svg>
    </div>
  );
};

export default TraceEditor;