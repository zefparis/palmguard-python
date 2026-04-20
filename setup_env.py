import os
os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'
os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'
os.environ['GALLIUM_DRIVER'] = 'softpipe'
os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.3'
os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'
os.environ['EGL_PLATFORM'] = 'surfaceless'
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
