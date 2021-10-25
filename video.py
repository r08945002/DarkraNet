import cv2
import os

def vid2frame(video_name):

    save_path = './frame_{}'.format(video_name[:-4])
    if not os.path.exists(save_path):
      os.makedirs(save_path)
    vidcap = cv2.VideoCapture(video_name)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    success,image = vidcap.read()
    count = 0
    while success:
        cv2.imwrite(os.path.join(save_path, 'frame_{}.jpg'.format(count)), image)     # save frame as JPEG file
        success,image = vidcap.read()
        count += 1

    return fps, save_path

def frame2vid(out_frame_path, video_name, fps):

    frame_n = len([name for name in os.listdir(out_frame_path) if os.path.isfile(os.path.join(out_frame_path, name))])
    img_array = []
    for n in range(frame_n):
        img = cv2.imread(os.path.join(out_frame_path, 'frame_{}.png'.format(n)))
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)

    out_vid = cv2.VideoWriter('output_{}.mp4'.format(video_name[:-4]), cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    for i in range(len(img_array)):
        out_vid.write(img_array[i])

    out_vid.release()
    print('-----> output_{}.mp4 is saved!'.format(video_name[:-4]))


if __name__=='__main__':

    video_name = 'WL.mp4'
    vid2frame(video_name)
