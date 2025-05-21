simple_emotions = {'Happy', 'Sad', 'Angry'};
complex_emotions = {'Pride', 'Guilt'};
all_emotions = [simple_emotions, complex_emotions];

video_files = {'Happy.mp4', 'Sad.mp4', 'Angry.mp4', 'Pride.mp4', 'Guilt.mp4'};
expected_emotions = {'Happy', 'Sad', 'Angry', 'Pride', 'Guilt'};
load('detected_emotions.mat'); 

fprintf('Performing EDA: Cleaning and validating detected emotions...\n');

valid_emotions = all_emotions;
cleaned_emotions = cell(size(detected_emotions));

for i = 1:length(detected_emotions)
    emotion = strtrim(detected_emotions{i}); 
    emotion = lower(emotion); % Normalize case
    emotion = regexprep(emotion, '[^a-z]', ''); 

    if ~isempty(emotion)
        emotion = [upper(emotion(1)) emotion(2:end)];
    end

    if ismember(emotion, valid_emotions)
        cleaned_emotions{i} = emotion;
    else
        fprintf('Warning: Noisy/Invalid emotion detected at index %d: "%s"\n', i, detected_emotions{i});
        cleaned_emotions{i} = 'Unknown';
    end
end

detected_emotions = cleaned_emotions;
correct_count = 0;
fprintf('\nAnalyzing videos for emotion classification...\n\n');

for i = 1:length(video_files)
    video_file = video_files{i};
    detected_emotion = detected_emotions{i};

    try
        v = VideoReader(video_file);
        fprintf('Video %d: %s\n', i, video_file);
    catch ME
        fprintf('Error reading video: %s\n', video_file);
        fprintf('MATLAB Error Message: %s\n', ME.message);
        continue;
    end

    if ismember(detected_emotion, simple_emotions)
        emotion_type = 'Simple';
    elseif ismember(detected_emotion, complex_emotions)
        emotion_type = 'Complex';
    else
        emotion_type = 'Unknown';
    end

    fprintf('Detected Emotion: %s\n', detected_emotion);
    fprintf('Emotion Type   : %s Emotion\n\n', emotion_type);

    if strcmpi(detected_emotion, expected_emotions{i})
        correct_count = correct_count + 1;
    end
    while hasFrame(v)
        frame = readFrame(v);
        imshow(frame, 'InitialMagnification', 'fit');
        title(sprintf('Video %d - Detected: %s (%s)', i, detected_emotion, emotion_type));
        drawnow;
        pause(1 / v.FrameRate);
    end
end

fprintf('\nEmotion classification completed.\n');
accuracy = (correct_count / length(video_files)) * 100;
fprintf('Classification Accuracy: %.2f%%\n', accuracy);

figure;
confusionchart(expected_emotions, detected_emotions);
title('Emotion Classification Confusion Matrix');
