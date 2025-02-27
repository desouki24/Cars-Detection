the_Video = VideoReader('Videos/traffic.avi');
Object_Detector = vision.ForegroundDetector('NumGaussians', 3, 'NumTrainingFrames', 50);

for i = 1:150 
frame = readFrame(the_Video);
the_Object = step(Object_Detector, frame);
end

figure; imshow(frame); title('Video Frame');
figure; imshow(the_Object); title('The Object');
Structure = strel('square', 3);
Noise_Free_Object = imopen(the_Object, Structure);
figure; 
imshow(Noise_Free_Object); title('Object After Removing Noise');
Bounding_Box = vision.BlobAnalysis( 'BoundingBoxOutputPort', true ,'AreaOutputPort', false, 'CentroidOutputPort', false,'MinimumBlobArea', 150);
the_Box = step(Bounding_Box, Noise_Free_Object);
Detected_Car = insertShape(frame, 'Rectangle', the_Box, 'Color', 'green');
Number_of_Cars = size(the_Box, 1);
Detected_Car = insertText(Detected_Car, [10 10], Number_of_Cars, 'BoxOpacity', 1,'FontSize', 14);
figure; 
imshow(Detected_Car); title('Detected Cars');
videoPlayer = vision.VideoPlayer('Name', 'Detected Cars');
videoPlayer.Position(3:4) = [650,400];

while hasFrame(the_Video)
frame = readFrame(the_Video); the_Object = step(Object_Detector, frame); Noise_Free_Object = imopen(the_Object, Structure); the_Box = step(Bounding_Box, Noise_Free_Object); Detected_Car = insertShape(frame, 'Rectangle', the_Box, 'Color', 'green'); Number_of_Cars = size(the_Box, 1); Detected_Car = insertText(Detected_Car, [10 10], Number_of_Cars, 'BoxOpacity', 1, 'FontSize', 14); step(videoPlayer, Detected_Car);
end
