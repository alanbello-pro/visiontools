import time

def print_progress(frame_count: int, total_frames: int, fps: float, num_unique_tracks: int, bar_length: int):


    if not hasattr(print_progress, "start_time"):
        print_progress.start_time = time.time()

    if frame_count == 0:
        print_progress.start_time = time.time()
        return

    percentual = (frame_count / total_frames) * 100
    
    filled_len = int(round(bar_length * frame_count / float(total_frames)))
    bar = '█' * filled_len + '-' * (bar_length - filled_len)
    tempo_decorrido = time.time() - print_progress.start_time
    minutos = int(tempo_decorrido // 60)
    segundos = int(tempo_decorrido % 60)
    fps_media = frame_count / (tempo_decorrido + 1e-6)

    print(
        f'\rProgresso: [{bar}] {percentual:.1f}% | ' 
        f'{frame_count}/{total_frames} | ' 
        f'FPS: {fps_media:.1f} | ' 
        f'Tempo: {minutos:02d}:{segundos:02d} | ' 
        f'Objetos: {num_unique_tracks}', 
        end='', flush=True
    )

