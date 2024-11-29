import ndarray from 'ndarray';
import { Tensor } from 'onnxruntime-web';
import ops from 'ndarray-ops';
import ObjectDetectionCamera from '../ObjectDetectionCamera';
import { round } from 'lodash';
// import { yoloClasses } from '../../data/yolo_classes';
import { yoloClasses } from '../../data/capstone_classes';
import { useState } from 'react';
import { useEffect } from 'react';
import { runModelUtils } from '../../utils';

const RES_TO_MODEL: [number[], string][] = [
  [[800, 800], 'ghost10_final.onnx']
];

const Yolo = (props: any) => {
  const [modelResolution, setModelResolution] = useState<number[]>(
    RES_TO_MODEL[0][0]
  );
  const [modelName, setModelName] = useState<string>(RES_TO_MODEL[0][1]);
  const [session, setSession] = useState<any>(null);
  const [showConfidence, setShowConfidence] = useState<boolean>(false);

  useEffect(() => {
    const getSession = async () => {
      const session = await runModelUtils.createModelCpu(
        `./_next/static/chunks/pages/${modelName}`
      );
      setSession(session);
    };
    getSession();
  }, [modelName, modelResolution]);

  const changeModelResolution = (width?: number, height?: number) => {
    if (width !== undefined && height !== undefined) {
      setModelResolution([width, height]);
      return;
    }
    const index = RES_TO_MODEL.findIndex((item) => item[0] === modelResolution);
    if (index === RES_TO_MODEL.length - 1) {
      setModelResolution(RES_TO_MODEL[0][0]);
      setModelName(RES_TO_MODEL[0][1]);
    } else {
      setModelResolution(RES_TO_MODEL[index + 1][0]);
      setModelName(RES_TO_MODEL[index + 1][1]);
    }
  };

  const resizeCanvasCtx = (
    ctx: CanvasRenderingContext2D,
    targetWidth: number,
    targetHeight: number,
    inPlace = false
  ) => {
    let canvas: HTMLCanvasElement;

    if (inPlace) {
      // Get the canvas element that the context is associated with
      canvas = ctx.canvas;

      // Set the canvas dimensions to the target width and height
      canvas.width = targetWidth;
      canvas.height = targetHeight;

      // Scale the context to the new dimensions
      ctx.scale(
        targetWidth / canvas.clientWidth,
        targetHeight / canvas.clientHeight
      );
    } else {
      // Create a new canvas element with the target dimensions
      canvas = document.createElement('canvas');
      canvas.width = targetWidth;
      canvas.height = targetHeight;

      // Draw the source canvas into the target canvas
      canvas
        .getContext('2d')!
        .drawImage(ctx.canvas, 0, 0, targetWidth, targetHeight);

      // Get a new rendering context for the new canvas
      ctx = canvas.getContext('2d')!;
    }

    return ctx;
  };

  const preprocess = (ctx: CanvasRenderingContext2D) => {
    const resizedCtx = resizeCanvasCtx(
      ctx,
      modelResolution[0],
      modelResolution[1]
    );

    const imageData = resizedCtx.getImageData(
      0,
      0,
      modelResolution[0],
      modelResolution[1]
    );
    const { data, width, height } = imageData;
    // data processing
    const dataTensor = ndarray(new Float32Array(data), [width, height, 4]);
    const dataProcessedTensor = ndarray(new Float32Array(width * height * 3), [
      1,
      3,
      width,
      height,
    ]);

    ops.assign(
      dataProcessedTensor.pick(0, 0, null, null),
      dataTensor.pick(null, null, 0)
    );
    ops.assign(
      dataProcessedTensor.pick(0, 1, null, null),
      dataTensor.pick(null, null, 1)
    );
    ops.assign(
      dataProcessedTensor.pick(0, 2, null, null),
      dataTensor.pick(null, null, 2)
    );

    ops.divseq(dataProcessedTensor, 255);

    const tensor = new Tensor('float32', new Float32Array(width * height * 3), [
      1,
      3,
      width,
      height,
    ]);

    (tensor.data as Float32Array).set(dataProcessedTensor.data);
    return tensor;
  };

  const conf2color = (conf: number) => {
    const r = Math.round(255 * (1 - conf));
    const g = Math.round(255 * conf);
    return `rgb(${r},${g},0)`;
  };

  // Define the color map
const color_map: { [key: number]: string } = {
  0: 'rgb(128, 0, 128)',   // purple
  1: 'rgb(0, 0, 255)',     // blue
  2: 'rgb(255, 255, 0)',   // yellow
  3: 'rgb(255, 0, 0)',     // red
  4: 'rgb(0, 255, 0)',     // lime
  5: 'rgb(255, 192, 203)', // pink
  6: 'rgb(0, 255, 255)',   // green
  7: 'rgb(0, 255, 255)',   // cyan
  8: 'rgb(255, 0, 255)',   // magenta
  9: 'rgb(0, 0, 0)',       // black
};

// Function to map cls_id to color
const clsIdToColor = (cls_id: number) => {
  // Return color if cls_id exists in color_map, otherwise default to white
  return color_map[cls_id] || 'rgb(255, 255, 255)';
};

const postprocess = async (
  tensor: Tensor,
  inferenceTime: number,
  ctx: CanvasRenderingContext2D,
  modelName: string
) => {
  return postprocessYolov10(ctx, modelResolution, tensor, clsIdToColor, showConfidence);
};

return (
  <ObjectDetectionCamera
    width={props.width}
    height={props.height}
    preprocess={preprocess}
    postprocess={postprocess}
    session={session}
    changeCurrentModelResolution={changeModelResolution}
    currentModelResolution={modelResolution}
    modelName={modelName}
    showConfidence={showConfidence}
    setShowConfidence={setShowConfidence}
  />
);
};
export default Yolo;

function nonMaximumSuppression(
  boxes: number[][],
  scores: number[],
  iouThreshold: number = 0.2
): number[][] {
  const keep: boolean[] = new Array(boxes.length).fill(true);

  // IoU calculation function
  const calculateIoU = (box1: number[], box2: number[]): number => {
    const x1 = Math.max(box1[0], box2[0]);
    const y1 = Math.max(box1[1], box2[1]);
    const x2 = Math.min(box1[2], box2[2]);
    const y2 = Math.min(box1[3], box2[3]);

    const interArea = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
    const box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1]);
    const box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1]);

    return interArea / (box1Area + box2Area - interArea);
  };

  // Sort boxes by confidence score in descending order
  const sortedIndices = scores
    .map((score, index) => ({ score, index }))
    .sort((a, b) => b.score - a.score)
    .map((item) => item.index);

  for (let i = 0; i < sortedIndices.length; i++) {
    const idx = sortedIndices[i];
    if (!keep[idx]) continue; // Skip if the box is already suppressed

    for (let j = i + 1; j < sortedIndices.length; j++) {
      const idx2 = sortedIndices[j];
      if (calculateIoU(boxes[idx], boxes[idx2]) > iouThreshold) {
        keep[idx2] = false; // Suppress box j if IoU > threshold
      }
    }
  }

  // Return the remaining boxes that were not suppressed
  return sortedIndices.filter((index) => keep[index]).map((index) => boxes[index]);
}

function postprocessYolov10(
  ctx: CanvasRenderingContext2D,
  modelResolution: number[],
  tensor: Tensor,
  clsIdToColor: (conf: number) => string,
  showConfidence: boolean
) {
  const dx = ctx.canvas.width / modelResolution[0];
  const dy = ctx.canvas.height / modelResolution[1];
  const detections = []; // Array to store detection logs
  const classCounts: Record<string, number> = {}; // Object to store class counts

  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);

  let x0, y0, x1, y1, cls_id, score;
  let boxes: number[][] = [];
  let scores: number[] = [];
  let classIds: number[] = [];

  // Collect the bounding boxes and scores
  for (let i = 0; i < tensor.dims[1]; i += 6) {
    [x0, y0, x1, y1, score, cls_id] = tensor.data.slice(i, i + 6);
    if ((score as any) < 0.5) {
      continue; // Skip low-confidence detections
    }

    // scale to canvas size
    [x0, x1] = [x0, x1].map((x: any) => x * dx);
    [y0, y1] = [y0, y1].map((x: any) => x * dy);

    [x0, y0, x1, y1, cls_id] = [x0, y0, x1, y1, cls_id].map((x: any) =>
      round(x)
    );
    [score] = [score].map((x: any) => round(x * 100, 1));

    const label =
      yoloClasses[cls_id].toString()[0] +
      yoloClasses[cls_id].toString().substring(1);

    // Update the class count
    if (classCounts[label]) {
      classCounts[label] += 1;
    } else {
      classCounts[label] = 1;
    }

    // Store the box, score, and classId for NMS
    boxes.push([x0, y0, x1, y1]);
    scores.push(score);
    classIds.push(cls_id);
  }

  // Apply Non-Maximum Suppression (NMS)
  const filteredBoxes = nonMaximumSuppression(boxes, scores);

  // Draw the filtered boxes
  for (const box of filteredBoxes) {
    const [x0, y0, x1, y1] = box;
    const cls_id = classIds[boxes.indexOf(box)];
    const score = scores[boxes.indexOf(box)];

    const label =
      yoloClasses[cls_id].toString()[0] +
      yoloClasses[cls_id].toString().substring(1);

    const color = clsIdToColor(cls_id);

    // Draw the bounding box
    // ctx.strokeStyle = color;
    // ctx.lineWidth = 3;
    // ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);

    // Calculate the center of the bounding box
    const centerX = (x0 + x1) / 2;
    const centerY = (y0 + y1) / 2;

    // Draw a dot at the center
    ctx.beginPath();
    ctx.arc(centerX, centerY, 3, 0, 2 * Math.PI); // Radius of 3 for the dot
    ctx.fillStyle = color;
    ctx.fill();
    ctx.closePath();
    
    const displayText = showConfidence ? `${label} -.${score}` : label;
    // Draw the label above the dot
    ctx.font = '20px Arial';
    ctx.fillStyle = color;
    ctx.textAlign = 'center'; // Center align the label
    ctx.fillText(displayText, centerX, centerY - 10); // Position label slightly above the dot

    // Add to detection log
    detections.push({
      className: label,
      confidence: score,
      // bbox: [x0, y0, x1 - x0, y1 - y0], // Optional bounding box data
    });
  }

  // Draw detection count in top left corner
  const totalCountText = `Total: ${Object.values(classCounts).reduce(
    (a, b) => a + b,
    0
  )}`;
  ctx.font = '30px Arial';
  const textWidth = ctx.measureText(totalCountText).width;
  const textHeight = 24;
  ctx.fillStyle = 'rgba(0, 0, 0, 0.6)';
  ctx.fillRect(25, 5, textWidth + 20, textHeight + 10);
  ctx.fillStyle = 'white';
  ctx.fillText(totalCountText, 100, 30);

  // Display individual class counts below the total count
  let offsetY = 60; // Start below the total count
  for (const [label, count] of Object.entries(classCounts)) {
    const classCountText = `Class ${label}: ${count}`;
    ctx.fillStyle = 'rgba(0, 0, 0, 0.6)';
    ctx.fillRect(25, offsetY - 20, ctx.measureText(classCountText).width + 10, 30);
    ctx.fillStyle = 'white';
    ctx.fillText(classCountText, 100, offsetY);
    offsetY += 45; // Move down for the next line
  }

  return detections;
}