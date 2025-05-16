import subprocess


class FFMPEGwriter:
    """Writer class that can add matplotlib figures as frames and then write gif or mp4
    using FFMPEG.

    """

    def __init__(self, fps=15):
        self.fps = fps
        self.frames = []
        self.width = None
        self.height = None

    def add_frame(self, fig):
        """Add matplotlib figure to frame list.

        Args:
            fig: matplotlib figure.

        """

        if not (self.width or self.height):
            self.width, self.height = fig.canvas.get_width_height()
        self.frames.append(
            fig.canvas.tostring_rgb()
        )  # extract the image as an RGB string

    def write(self, filename):
        """Write frames to gif or mp4 using FFMPEG.

        Args:
            filename: name of output file. File ending must be ".gif" or ".mp4".

        """

        if ".gif" in filename:
            palette_cmd = (
                "ffmpeg",
                "-s",
                "{}x{}".format(self.width, self.height),
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-i",
                "-",
                "-filter_complex",
                "palettegen=stats_mode=diff",
                "-y",
                "palette.png",
                "-hide_banner",  # less verbose
                "-loglevel",
                "error",
            )
            palette_process = subprocess.Popen(palette_cmd, stdin=subprocess.PIPE)
            for frame in self.frames:
                palette_process.stdin.write(frame)  # write frame to GIF palette
            palette_process.communicate()  # Create palette
            animation_cmd = (
                "ffmpeg",
                "-y",  # overwrite output file
                "-r",
                str(self.fps),  # frame rate
                "-s",
                "{}x{}".format(self.width, self.height),  # size of image string
                "-pix_fmt",
                "rgb24",  # input format
                "-f",
                "rawvideo",
                "-thread_queue_size",
                "512",
                "-i",
                "-",  # tell ffmpeg to expect raw video from the pipe
                "-thread_queue_size",
                "512",
                "-i",
                "palette.png",
                "-filter_complex",
                "paletteuse",
                "-vframes",
                str(len(self.frames)),  # number of frames
                filename,
                "-hide_banner",  # less verbose
                "-loglevel",
                "error",
            )  # file name
        elif ".mp4" in filename:
            animation_cmd = (
                "ffmpeg",
                "-y",  # overwrite output file
                "-r",
                str(self.fps),  # frame rate
                "-s",
                "{}x{}".format(self.width, self.height),  # size of image string
                "-pix_fmt",
                "rgb24",  # input format
                "-f",
                "rawvideo",
                "-i",
                "-",  # tell ffmpeg to expect raw video from the pipe
                "-vcodec",
                "h264",  # output encoding
                "-pix_fmt",
                "yuv420p",  # required for some media players
                "-vframes",
                str(len(self.frames)),  # number of frames
                filename,
                "-hide_banner",  # less verbose
                "-loglevel",
                "error",
            )  # file name
        else:
            raise Exception('FFMPEGwriter expects ".gif" or ".mp4" in filename')
        animation_process = subprocess.Popen(animation_cmd, stdin=subprocess.PIPE)
        for frame in self.frames:
            animation_process.stdin.write(frame)  # write frame to animation
        animation_process.communicate()  # Create animation
