def build_vsf_command(path, video_path, output_dir, subtitle_path,
                      top_end, bottom_end, left_end, right_end,
                      cpu_count, decoder, use_cuda=False,
                      include_ocr_threads=False):
    """Build a shell-free VideoSubFinder command line."""
    command = [path, '-c', '-r', '-i', video_path,
               '-o', output_dir, '-ces', subtitle_path]
    if use_cuda:
        command.append('--use_cuda')
    command.extend([
        '-te', str(top_end), '-be', str(bottom_end),
        '-le', str(left_end), '-re', str(right_end),
        '-nthr', str(cpu_count),
    ])
    if include_ocr_threads:
        command.extend(['-nocrthr', str(cpu_count)])
    command.append(f'--open_video_{decoder.lower()}')
    return command
