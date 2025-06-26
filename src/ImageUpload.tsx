import React, { useState, useEffect } from 'react';
import { useDropzone } from 'react-dropzone';
import ContourEditor from './ContourEditor';
import MaskPainter from './MaskPainter';
import TraceEditor from './TraceEditor';
import './ImageUpload.css';

// interface Trace {
//     start: [number, number];
//     end: [number, number];
// }

type Point = [number, number];
type Polygon = Point[];
type TraceData = { [id: string]: Polygon };

const ImageUpload: React.FC = () => {
  const [image, setImage] = useState<string | null>(null);       // Local preview
  const [file, setFile] = useState<File | null>(null);           // Uploaded file
  
  type Mode = 'edit' | 'flex' | null;
  
  const [mode, setMode] = useState<Mode>(null);
  const [contourImage, setContourImage] = useState<string | null>(null);
  const [editedImage, setEditedImage] = useState<string | null>(null);
  const [imageID, setImageID] = useState<string | null>(null); // Store the unique image_id
  const [contourPoints, setContourPoints] = useState<[number, number][]>([]);
  const [isEditing, setIsEditing] = useState<boolean>(false);
  const [maskImage, setMaskImage]  = useState<string | null>(null);
  const [tracedImage, setTracedImage]  = useState<string | null>(null);
  const [traceId, setTraceId] = useState<string | null>(null);
//   const [traces, setTraces] = useState<[number, number][][]>([]);
    const [traces, setTraces] = useState<TraceData>({});
    const [finalOutline, setFinalOutline] = useState<Polygon | null>(null);

  const { getRootProps, getInputProps } = useDropzone({
    accept: {
      'image/*': ['.jpeg', '.png', '.gif', '.jpg', '.webp'],
    },
    onDrop: (acceptedFiles: File[]) => {
      const selectedFile = acceptedFiles[0];
      setFile(selectedFile);
      setImage(URL.createObjectURL(selectedFile));
      setImageID(null);
      setContourImage(null); // Reset contour image to be hidden
      setMaskImage(null) // Reset mask image to be hidden
      setTraceId(null);
      setTracedImage(null);
      setTraces({});
    },
  });

  const handleConfirm = async () => {
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);  // Match FastAPI param name

    try {
      const response = await fetch('http://localhost:8000/upload', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      console.log('Contour response:', data);
      setContourImage(`http://localhost:8000/${data.contour_image}`); // Collect contour image data
      setImageID(data.image_id);
      setContourPoints(data.contour_points); // Collect contour points data
      setImage(null); // Hide original once confirmed
    } catch (err) {
      console.error('Error uploading image:', err);
    }
  };

  const handleFinishContour = async () => {
    try {
        const response = await fetch(`http://localhost:8000/api/save-contour/${imageID}`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          contour: contourPoints, // array of [x, y]
        //   image_id: imageID,
        }),
      });
  
      const data = await response.json();

      console.log("Mask saved:", data);
      setMaskImage(`http://localhost:8000/${data.mask_path}`)
      setContourImage(null); // Hide contour image
      setIsEditing(false);
    } catch (error) {
      console.error("Error saving contour:", error);
    }
  };
    
    
  const handleSaveEditedTraces = async () => {
    if (!imageID || Object.keys(traces).length === 0) {
      alert("No edited traces to save. Please generate and edit traces first.");
      return;
    }

    try {
      const response = await fetch(`http://localhost:8000/save-edited-traces/${imageID}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        // The body now contains an object with a "polygons" key,
        // which holds our traces object, matching the Pydantic model.
        body: JSON.stringify({ polygons: traces }),
      });
      const data = await response.json();
      console.log('Save edited traces response:', data);
      alert(data.message);
    } catch (err) {
      console.error('Error saving edited traces:', err);
      alert('Failed to save edited traces.');
    }
  };
    
      // New function to generate KiCad file
    const handleGenerateKiCad = async () => {
        if (!imageID) {
            alert("No image ID available to generate KiCad file. Please upload and process an image first.");
            return;
        }

        try {
            const response = await fetch(`http://localhost:8000/generate-kicad/${imageID}`, {
            method: 'POST',
            });

            if (response.ok) {
            const blob = await response.blob();
            const contentDisposition = response.headers.get('Content-Disposition');
            let filename = `${imageID}_kicad_pcb.zip`;  // Default filename
            if (contentDisposition) {
                const filenameMatch = contentDisposition.match(/filename="(.+)"/);
                if (filenameMatch && filenameMatch[1]) {
                filename = filenameMatch[1];
                console.log("New filename!")
                console.log(filenameMatch[1])
                }
            }

            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            a.remove();
            window.URL.revokeObjectURL(url);
            alert("KiCad file generated and downloaded successfully!");
            } else {
            const errorData = await response.json();
            alert(`Failed to generate KiCad file: ${errorData.detail}`);
            }
        } catch (err) {
            console.error('Error generating KiCad file:', err);
            alert('Failed to generate KiCad file.');
        }
    };

//   // This useEffect would be triggered once a contour is confirmed to automatically proceed to mask painting
//   useEffect(() => {
//     if (contourPoints.length > 0 && contourImage && imageID) {
//         // You would typically call a backend endpoint here to create the mask
//         // and then pass the mask image URL to MaskPainter.
//         // For now, using the mask creation logic from the backend's /api/save-contour
//         // and assuming it's available or integrating it directly if needed.

//         // To create a mask, we need to send the contour points back to the backend
//         // This is the /api/save-contour endpoint
//         const createMask = async () => {
//             try {
//                 const response = await fetch('http://localhost:8000/api/save-contour', {
//                     method: 'POST',
//                     headers: {
//                         'Content-Type': 'application/json',
//                     },
//                     body: JSON.stringify({
//                         contour: contourPoints,
//                         image_id: `output/${imageID}_cropped_hand.png` // Use the correct cropped image path
//                     }),
//                 });
//                 const data = await response.json();
//                 console.log('Mask creation response:', data);
//                 setMaskImage(`http://localhost:8000/${data.mask_path}`);
//                 setMode('edit'); // Assuming 'edit' mode refers to mask painting
//             } catch (err) {
//                 console.error('Error creating mask:', err);
//                 alert('Failed to create mask.');
//             }
//         };
//         createMask();
//     }
//   }, [contourPoints, contourImage, imageID]);

  return (
    <div>
      <div {...getRootProps({ className: 'dropzone' })}>
        <input {...getInputProps()} />
        <p>Drag & drop an image here, or click to select one</p>
      </div>

      <br />

      {image && (
        <>
          <img src={image} alt="Uploaded Hand" width="300" />
          <br />
          <button onClick={handleConfirm} className="mt-2 bg-blue-500 text-white px-4 py-2 rounded">
            Confirm Image
          </button>
        </>
      )}


    {contourImage && !isEditing && (
      <>
        <h4>Detected Contours:</h4>
        <img src={contourImage} alt="Contour Result" width="300" />
        <br />
        <button onClick={() => setIsEditing(true)} className="mt-2 bg-green-500 text-white px-4 py-2 rounded">
          Edit Contour
        </button>
      </>
    )}

    {/* Contour editing mode */}
    {contourImage && isEditing && (
        <>
            <ContourEditor
                imageUrl={contourImage}
                initialContour={contourPoints}
                onUpdate={(updatedPoints) => {
                    console.log('Updated contour points:', updatedPoints);
                    setContourPoints(updatedPoints)
                    // Optionally save or pass the updated points back up here
                }}
            />
            <button onClick={handleFinishContour} className="mt-2 bg-blue-500 text-white px-4 py-2 rounded">
            Finish Contour
            </button>
        </>
    )}

    {maskImage && !tracedImage && (
        <>
            <MaskPainter
                imageUrl={maskImage}
                width={500}
                imageID={imageID}
                onExport={async (traceData) => {
                    // setTraceId(traceData.trace_id);
                    // setTracedImage(`http://localhost:8000/${traceData.overlay_path}`);
                    setTraceId(traceData.trace_id);
                    setTracedImage(`http://localhost:8000/${traceData.overlay_path}`);
                    // Filter out any empty/null regions from the backend
                    const validPolygons: TraceData = {};
                    for (const key in traceData.polygons) {
                    if (traceData.polygons[key] && traceData.polygons[key].length > 0) {
                        validPolygons[key] = traceData.polygons[key];
                    }
                    }
                    setTraces(validPolygons);
                    // After getting the traces, fetch the final board outline
                    if (imageID) {
                        try {
                        const outlineResponse = await fetch(`http://localhost:8000/get-final-outline/${imageID}`);
                        if (outlineResponse.ok) {
                            const outlineData = await outlineResponse.json();
                            setFinalOutline(outlineData.outline);
                        } else {
                            console.error("Failed to fetch final outline:", await outlineResponse.text());
                        }
                        } catch (err) {
                        console.error("Error fetching final outline:", err);
                        }
                    }
                  }}
            />
        </>
    )}

    {/* {traceId && tracedImage && traces.length > 0 && (
        <>
          <h4>Edit Traces:</h4>
          <TraceEditor
            traceId={traceId}
            imageUrl={tracedImage}
            width={500}
            initialTraces={traces}
            onChange={(updated: [number, number][][]) => setTraces(updated)}
          />
          {/* <button onClick={() => saveTracesToServer(traces)}>Save</button> */}
        {/* </>
    )} */}

    {tracedImage && traces && (
        <>
            <TraceEditor
                tracedImage={tracedImage}
                initialTraces={traces}
                finalOutline={finalOutline}
                onChange={(updatedTraces) => {
                    setTraces(updatedTraces); // Optional: save updated polygons
                }}
                width={500}
            />
            <button onClick={handleSaveEditedTraces} className="action-button">Save Edited Traces</button>
            <button onClick={handleGenerateKiCad} className="action-button">Generate KiCad File</button>
        </>
      )}
    </div>
  );
};

export default ImageUpload;