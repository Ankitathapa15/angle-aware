import { useEffect, useRef, useState } from 'react';
import * as poseDetection from '@tensorflow-models/pose-detection';
import '@tensorflow/tfjs-backend-webgl';
import { Button } from './ui/button';
import { Card } from './ui/card';

interface PostureStatus {
  score: number;
  message: string;
  color: 'success' | 'warning' | 'destructive';
}

export const PostureDetector = () => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isActive, setIsActive] = useState(false);
  const [postureStatus, setPostureStatus] = useState<PostureStatus>({
    score: 0,
    message: 'Start camera to begin',
    color: 'success',
  });
  const detectorRef = useRef<poseDetection.PoseDetector | null>(null);
  const animationRef = useRef<number>();

  // Initialize pose detector
  const initDetector = async () => {
    const model = poseDetection.SupportedModels.MoveNet;
    const detectorConfig = {
      modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING,
    };
    detectorRef.current = await poseDetection.createDetector(model, detectorConfig);
  };

  // Calculate angle between three points
  const calculateAngle = (point1: { x: number; y: number }, point2: { x: number; y: number }, point3: { x: number; y: number }) => {
    const radians = Math.atan2(point3.y - point2.y, point3.x - point2.x) - Math.atan2(point1.y - point2.y, point1.x - point2.x);
    let angle = Math.abs((radians * 180.0) / Math.PI);
    if (angle > 180.0) {
      angle = 360 - angle;
    }
    return angle;
  };

  // Analyze posture and calculate score
  const analyzePosture = (keypoints: poseDetection.Keypoint[]): PostureStatus => {
    const leftShoulder = keypoints.find(kp => kp.name === 'left_shoulder');
    const rightShoulder = keypoints.find(kp => kp.name === 'right_shoulder');
    const leftHip = keypoints.find(kp => kp.name === 'left_hip');
    const rightHip = keypoints.find(kp => kp.name === 'right_hip');

    if (!leftShoulder || !rightShoulder || !leftHip || !rightHip) {
      return { score: 0, message: 'Position yourself in frame', color: 'warning' };
    }

    // Check confidence scores
    const minConfidence = 0.3;
    if (
      leftShoulder.score! < minConfidence ||
      rightShoulder.score! < minConfidence ||
      leftHip.score! < minConfidence ||
      rightHip.score! < minConfidence
    ) {
      return { score: 0, message: 'Move closer or improve lighting', color: 'warning' };
    }

    // Calculate shoulder tilt
    const shoulderDiff = Math.abs(leftShoulder.y - rightShoulder.y);
    const shoulderWidth = Math.abs(leftShoulder.x - rightShoulder.x);
    const shoulderTilt = (shoulderDiff / shoulderWidth) * 100;

    // Calculate hip tilt
    const hipDiff = Math.abs(leftHip.y - rightHip.y);
    const hipWidth = Math.abs(leftHip.x - rightHip.x);
    const hipTilt = (hipDiff / hipWidth) * 100;

    // Calculate spine alignment
    const shoulderCenter = { x: (leftShoulder.x + rightShoulder.x) / 2, y: (leftShoulder.y + rightShoulder.y) / 2 };
    const hipCenter = { x: (leftHip.x + rightHip.x) / 2, y: (leftHip.y + rightHip.y) / 2 };
    const spineAngle = Math.abs(90 - Math.abs(Math.atan2(hipCenter.y - shoulderCenter.y, hipCenter.x - shoulderCenter.x) * 180 / Math.PI));

    // Calculate score (0-100)
    let score = 100;
    score -= shoulderTilt * 2; // Penalize shoulder tilt
    score -= hipTilt * 2; // Penalize hip tilt
    score -= spineAngle * 0.5; // Penalize spine misalignment
    score = Math.max(0, Math.min(100, score));

    // Determine feedback
    if (shoulderTilt > 15) {
      return { score: Math.round(score), message: 'Straighten your shoulders', color: 'destructive' };
    } else if (hipTilt > 15) {
      return { score: Math.round(score), message: 'Adjust your sitting posture', color: 'warning' };
    } else if (score > 80) {
      return { score: Math.round(score), message: 'Good posture detected!', color: 'success' };
    } else if (score > 60) {
      return { score: Math.round(score), message: 'Minor adjustments needed', color: 'warning' };
    } else {
      return { score: Math.round(score), message: 'Poor posture - sit up straight', color: 'destructive' };
    }
  };

  // Draw pose on canvas
  const drawPose = (poses: poseDetection.Pose[], ctx: CanvasRenderingContext2D) => {
    if (!poses || poses.length === 0) return;

    const pose = poses[0];
    const keypoints = pose.keypoints;

    // Draw keypoints
    keypoints.forEach(keypoint => {
      if (keypoint.score! > 0.3) {
        ctx.beginPath();
        ctx.arc(keypoint.x, keypoint.y, 5, 0, 2 * Math.PI);
        ctx.fillStyle = '#3B82F6';
        ctx.fill();
      }
    });

    // Draw connections
    const connections = [
      ['left_shoulder', 'right_shoulder'],
      ['left_hip', 'right_hip'],
      ['left_shoulder', 'left_hip'],
      ['right_shoulder', 'right_hip'],
    ];

    connections.forEach(([start, end]) => {
      const startPoint = keypoints.find(kp => kp.name === start);
      const endPoint = keypoints.find(kp => kp.name === end);

      if (startPoint && endPoint && startPoint.score! > 0.3 && endPoint.score! > 0.3) {
        ctx.beginPath();
        ctx.moveTo(startPoint.x, startPoint.y);
        ctx.lineTo(endPoint.x, endPoint.y);
        ctx.strokeStyle = '#8B5CF6';
        ctx.lineWidth = 3;
        ctx.stroke();
      }
    });
  };

  // Main detection loop
  const detectPose = async () => {
    if (!videoRef.current || !canvasRef.current || !detectorRef.current || !isActive) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    if (!ctx) return;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    try {
      const poses = await detectorRef.current.estimatePoses(video);
      
      if (poses.length > 0) {
        drawPose(poses, ctx);
        const status = analyzePosture(poses[0].keypoints);
        setPostureStatus(status);
      }
    } catch (error) {
      console.error('Pose detection error:', error);
    }

    animationRef.current = requestAnimationFrame(detectPose);
  };

  // Start camera
  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 },
      });

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.onloadedmetadata = () => {
          videoRef.current?.play();
          if (canvasRef.current && videoRef.current) {
            canvasRef.current.width = videoRef.current.videoWidth;
            canvasRef.current.height = videoRef.current.videoHeight;
          }
        };
      }

      if (!detectorRef.current) {
        await initDetector();
      }

      setIsActive(true);
    } catch (error) {
      console.error('Camera access error:', error);
      setPostureStatus({ score: 0, message: 'Camera access denied', color: 'destructive' });
    }
  };

  // Stop camera
  const stopCamera = () => {
    const stream = videoRef.current?.srcObject as MediaStream;
    stream?.getTracks().forEach(track => track.stop());
    if (videoRef.current) videoRef.current.srcObject = null;
    if (animationRef.current) cancelAnimationFrame(animationRef.current);
    setIsActive(false);
    setPostureStatus({ score: 0, message: 'Camera stopped', color: 'success' });
  };

  useEffect(() => {
    if (isActive) {
      detectPose();
    }
    return () => {
      if (animationRef.current) cancelAnimationFrame(animationRef.current);
    };
  }, [isActive]);

  const getScoreColor = (score: number) => {
    if (score >= 80) return 'text-success';
    if (score >= 60) return 'text-warning';
    return 'text-destructive';
  };

  const getMessageColor = (color: 'success' | 'warning' | 'destructive') => {
    const colors = {
      success: 'text-success',
      warning: 'text-warning',
      destructive: 'text-destructive',
    };
    return colors[color];
  };

  return (
    <div className="flex flex-col items-center gap-6 w-full max-w-4xl mx-auto p-6">
      <div className="text-center space-y-2">
        <h1 className="text-4xl font-bold bg-gradient-primary bg-clip-text text-transparent">
          AI Posture Corrector
        </h1>
        <p className="text-muted-foreground">
          Real-time posture analysis using AI-powered pose detection
        </p>
      </div>

      <Card className="relative overflow-hidden bg-card/50 backdrop-blur-sm border-border">
        <div className="relative">
          <video
            ref={videoRef}
            className="w-full h-auto rounded-t-lg"
            style={{ transform: 'scaleX(-1)' }}
          />
          <canvas
            ref={canvasRef}
            className="absolute top-0 left-0 w-full h-full"
            style={{ transform: 'scaleX(-1)' }}
          />
        </div>

        <div className="p-6 space-y-4 bg-gradient-to-t from-card to-transparent">
          <div className="flex items-center justify-between">
            <div className="space-y-1">
              <p className="text-sm text-muted-foreground">Posture Score</p>
              <p className={`text-5xl font-bold ${getScoreColor(postureStatus.score)}`}>
                {postureStatus.score}
              </p>
            </div>
            <div className="text-right space-y-1">
              <p className="text-sm text-muted-foreground">Status</p>
              <p className={`text-xl font-semibold ${getMessageColor(postureStatus.color)}`}>
                {postureStatus.message}
              </p>
            </div>
          </div>

          <Button
            onClick={isActive ? stopCamera : startCamera}
            className="w-full"
            variant={isActive ? 'destructive' : 'default'}
            size="lg"
          >
            {isActive ? 'Stop Camera' : 'Start Camera'}
          </Button>
        </div>
      </Card>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 w-full">
        <Card className="p-4 bg-card/50 backdrop-blur-sm">
          <div className="space-y-2">
            <div className="w-10 h-10 rounded-full bg-success/20 flex items-center justify-center">
              <div className="w-5 h-5 rounded-full bg-success"></div>
            </div>
            <h3 className="font-semibold">Good Posture</h3>
            <p className="text-sm text-muted-foreground">Score above 80 indicates excellent alignment</p>
          </div>
        </Card>

        <Card className="p-4 bg-card/50 backdrop-blur-sm">
          <div className="space-y-2">
            <div className="w-10 h-10 rounded-full bg-warning/20 flex items-center justify-center">
              <div className="w-5 h-5 rounded-full bg-warning"></div>
            </div>
            <h3 className="font-semibold">Minor Issues</h3>
            <p className="text-sm text-muted-foreground">Score 60-80 needs small adjustments</p>
          </div>
        </Card>

        <Card className="p-4 bg-card/50 backdrop-blur-sm">
          <div className="space-y-2">
            <div className="w-10 h-10 rounded-full bg-destructive/20 flex items-center justify-center">
              <div className="w-5 h-5 rounded-full bg-destructive"></div>
            </div>
            <h3 className="font-semibold">Poor Posture</h3>
            <p className="text-sm text-muted-foreground">Score below 60 requires correction</p>
          </div>
        </Card>
      </div>
    </div>
  );
};
