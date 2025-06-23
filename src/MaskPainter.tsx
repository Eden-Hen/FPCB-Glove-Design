import React, { useEffect, useRef, useState, useImperativeHandle, forwardRef } from 'react';
import { Stage, Layer, Image as KonvaImage, Line, Circle } from 'react-konva';
import useImage from 'use-image';

interface Stroke {
  points: number[][];
  brushSize: number;
}

type Point = [number, number];
type Polygon = Point[];


interface MaskPainterProps {
  imageUrl: string;
  width?: number;
  imageID: string | null;
  onExport?: (traceData: { trace_id: string; overlay_path: string; polygons: Polygon[] }) => void;
}

const MaskPainter: React.FC<MaskPainterProps> = ({
  imageUrl,
  width = 500,
  imageID,
  onExport
}) => {
    const [image] = useImage(imageUrl, 'anonymous'); // The 'anonymous' is for CORS
  const [canvasHeight, setCanvasHeight] = useState<number | null>(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [strokes, setStrokes] = useState<Stroke[]>([]);
  const [currentStroke, setCurrentStroke] = useState<number[][]>([]);
  const [cursorPos, setCursorPos] = useState<{ x: number; y: number } | null>(null);
  const [brushSize, setBrushSize] = useState(10);
  const stageRef = useRef<any>(null);
  const [exportedImageUrl, setExportedImageUrl] = useState<string | null>(null);


  useEffect(() => {
    if (image) {
      const scale = width / image.width;
      setCanvasHeight(image.height * scale);
    }
  }, [image, width]);

  const handleMouseDown = (e: any) => {
    const pos = e.target.getStage().getPointerPosition();
    if (pos) {
        setCursorPos(pos);
        setCurrentStroke([[pos.x, pos.y]]);
        setIsDrawing(true);
    }
  };

  const handleMouseMove = (e: any) => {
    const pos = e.target.getStage().getPointerPosition();
    if (pos) {
        setCursorPos(pos);
        if(isDrawing){
            setCurrentStroke(prev => [...prev, [pos.x, pos.y]]);
        }
    }
  };

  const handleMouseUp = () => {
    if (currentStroke.length > 0) {
        const finalizedStroke =
          currentStroke.length === 1
            ? [currentStroke[0], currentStroke[0]] // Duplicate the point to draw a dot
            : currentStroke;
      
        setStrokes(prev => [...prev, { points: finalizedStroke, brushSize }]);
        setCursorPos(null);
        setCurrentStroke([]);
    }
    setIsDrawing(false);

    // // Export the drawn mask
    // if (onExport && stageRef.current) {
    //   const dataUrl = stageRef.current.toDataURL({ pixelRatio: 1 });
    //   onExport(dataUrl);
    // }
  };

  const handleUndo = () => {
    setStrokes(prev => prev.slice(0, -1));
  };

//   const handleExport = () => {
//     if (!stageRef.current || !image) {
//       alert('Stage or image not ready');
//       return;
//     }
  
//     const pixelRatio = image.width / width;
//     const canvas = stageRef.current.toCanvas({ pixelRatio });
  
//     canvas.toBlob(
//       async (blob: Blob | null) => {
//         if (!blob) {
//           alert('Failed to generate blob.');
//           return;
//         }
  
//         const formData = new FormData();
//         formData.append('file', blob, 'hand_mask.png');
  
//         try {
//           const response = await fetch('http://localhost:8000/upload-mask', {
//             method: 'POST',
//             body: formData,
//           });
  
//           const result = await response.json();
//           console.log('Mask upload success:', result);
  
//           // Trigger trace generation
//           const traceRes = await fetch('http://localhost:8000/generate-traces', {
//             method: 'POST',
//             headers: {
//               'Content-Type': 'application/json',
//             },
//             body: JSON.stringify({
//               image_id: result.image_id,  // Comes from upload-mask response
//             }),
//           });
  
//           const traceData = await traceRes.json();
//           console.log('Traces generated:', traceData);
  
//           alert('Traces generated!');
          
//           // Render exported image within MaskPainter
//           {exportedImageUrl && (
//             <>
//               <h4>Generated PCB Trace:</h4>
//               <img src={exportedImageUrl} alt="Trace result" width="300" />
//             </>
//           )}
  
//         } catch (error) {
//           console.error('Error:', error);
//           alert('Failed during mask upload or trace generation.');
//         }
//       },
//       'image/png'
//     );
//   };

const handleExport = () => {
    if (!stageRef.current || !image) {
      alert('Stage or image not ready');
      return;
    }
    if (!imageID) { // Check if imageID is available
        alert('Image ID is not available. Please start from image upload.');
        return;
    }
  
    const pixelRatio = image.width / width;
    const canvas = stageRef.current.toCanvas({ pixelRatio });
  
    canvas.toBlob(
      async (blob: Blob | null) => {
        if (!blob) {
          alert('Failed to generate blob.');
          return;
        }
  
        const formData = new FormData();
        formData.append('file', blob, `${imageID}_mask.png`);
  
        try {
          // Step 1: Upload the binary mask
          const response = await fetch(`http://localhost:8000/upload-mask/${imageID}`, {
            method: 'POST',
            body: formData,
          });
  
          const result = await response.json();
          console.log('Mask upload success:', result);
  
          // Step 2: Generate traces using uploaded image ID
          const traceRes = await fetch(`http://localhost:8000/generate-traces/${imageID}`, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({
              image_id: result.image_id,
            }),
          });
  
          const traceData = await traceRes.json();
          console.log('Traces generated:', traceData);
  
          // Step 3: Pass trace metadata to parent
          if(onExport){
            onExport({
                trace_id: traceData.trace_id,
                overlay_path: traceData.overlay_path,
                polygons: traceData.polygons
              });
          }
  
        } catch (error) {
          console.error('Error during export:', error);
          alert('Failed during mask upload or trace generation.');
        }
      },
      'image/png'
    );
  };
  

  if (!image || canvasHeight === null) return <p>Loading image...</p>;

  return (
    <div>
      <div className="mb-2 flex gap-4 items-center">
        <label>
          Brush Size:
          <input
            type="range"
            min={2}
            max={50}
            value={brushSize}
            onChange={(e) => setBrushSize(Number(e.target.value))}
            className="ml-2"
          />
        </label>
        <button
          onClick={handleUndo}
          className="px-3 py-1 bg-red-500 text-white rounded"
        >
          Undo
        </button>
      </div>

      <Stage
        ref={stageRef}
        width={width}
        height={canvasHeight}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        style={{ border: '1px solid #ccc', cursor: 'crosshair' }}
      >
        <Layer>
          <KonvaImage image={image} width={width} height={canvasHeight} />

          {/* Render all previous strokes with their individual brush sizes */}
          {strokes.map((stroke, i) => (
            <Line
              key={i}
              points={stroke.points.flat()}
              stroke="black"
              strokeWidth={stroke.brushSize}
              lineCap="round"
              lineJoin="round"
              tension={0.5}
            />
          ))}

          {/* Render current stroke with the current brush size */}
          {isDrawing && (
            <Line
              points={currentStroke.flat()}
              stroke="black"
              strokeWidth={brushSize}
              lineCap="round"
              lineJoin="round"
              tension={0.5}
            />
          )}

          {/* Visual indicator of brush radius */}
          {cursorPos && (
            <Circle
              x={cursorPos.x}
              y={cursorPos.y}
              radius={brushSize/2}
              stroke="rgba(0,0,255,0.5)"
              strokeWidth={1}
              dash={[4, 4]}
              listening={false}
            />
          )}
        </Layer>
      </Stage>
      <button
            onClick={handleExport}
            className="mt-4 bg-purple-600 text-white px-4 py-2 rounded"
            >
            Finish Mask
        </button>
    </div>
  );
};

export default MaskPainter;