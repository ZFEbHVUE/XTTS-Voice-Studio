#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XTTS Voice Studio — Tkinter GUI
Graphical interface for all XTTS-Voice-Studio scripts.
"""

import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
import subprocess
import threading
import sys
import os

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_root = None

# ── Cross-tab hand-off ───────────────────────────────────────────────────────
# HANDOFF holds the latest result blocks captured from a tool's stdout; TARGETS
# maps names to the destination StringVars (registered by each tab as it builds).
# A hand-off button copies HANDOFF -> the target field, so results flow
# Analyser -> Validator -> Comparator without copy/paste.
HANDOFF = {'xtts': '', 'audio': '', 'win_xtts': ''}
TARGETS = {}

def _handoff_set(target_name, value):
    sv = TARGETS.get(target_name)
    if sv is not None and value:
        sv.set(value.strip())
        return True
    return False


# Default directories

XTTS_ROOT     = os.path.dirname(SCRIPTS_DIR)  # parent of Python_Scripting/
DIR_PROMPTS   = os.path.join(XTTS_ROOT, "Prompts")
DIR_OUTPUT    = os.path.join(XTTS_ROOT, "Output_Song_files")
DIR_VOICES    = os.path.join(XTTS_ROOT, "Voices_Cloning")
DIR_AMBIENT   = os.path.join(XTTS_ROOT, "Ambient_Musics")
DIR_PUNCTUAL  = os.path.join(XTTS_ROOT, "Punctual_sounds")
DIR_MP3       = os.path.join(XTTS_ROOT, "MP3toTXT")
DIR_TXT       = os.path.join(XTTS_ROOT, "Song_to_TXT_with_Pauses")

def _ensure_dir(d):
    """Return directory if it exists, otherwise HOME."""
    return d if os.path.isdir(d) else os.path.expanduser("~")

def browse_file(var, filetypes=None, save=False, initialdir=None):
    if filetypes is None:
        filetypes = [("All", "*.*")]
    d = _ensure_dir(initialdir) if initialdir else None
    if save:
        path = filedialog.asksaveasfilename(filetypes=filetypes,
                                            initialdir=d)
    else:
        path = filedialog.askopenfilename(filetypes=filetypes,
                                          initialdir=d)
    if path:
        var.set(path)

def browse_files(var, filetypes=None, initialdir=None):
    if filetypes is None:
        filetypes = [("All", "*.*")]
    d = _ensure_dir(initialdir) if initialdir else None
    paths = filedialog.askopenfilenames(filetypes=filetypes,
                                        initialdir=d)
    if paths:
        var.set(" ".join(paths))

# Global audio player (one at a time)
_player_state = {'proc': None, 'btn': None}

def _stop_player():
    """Stop current player and reset button."""
    if _player_state['proc'] is not None:
        try:
            _player_state['proc'].terminate()
            _player_state['proc'].kill()
        except Exception:
            pass
        _player_state['proc'] = None
    if _player_state['btn'] is not None:
        try:
            _player_state['btn'].config(text="> Play", bg='#1a6b9e')
        except Exception:
            pass
        _player_state['btn'] = None

def play_toggle(path, btn=None):
    """Toggle Play/Stop for audio file."""
    # If active button → stop
    if _player_state['btn'] is btn and btn is not None:
        _stop_player()
        return
    # Otherwise stop current and start new
    _stop_player()

    if not path:
        return
    p = path.strip()
    if not p or not os.path.exists(p):
        return

    try:
        proc = subprocess.Popen(
            ['ffplay', '-nodisp', '-autoexit', '-loglevel', 'quiet', p],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        try:
            proc = subprocess.Popen(['aplay', p],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except FileNotFoundError:
            return

    _player_state['proc'] = proc
    _player_state['btn']  = btn
    if btn:
        btn.config(text="[] Stop", bg='#c0392b')

    def _watch():
        proc.wait()
        if _player_state['proc'] is proc:
            _player_state['proc'] = None
            _player_state['btn']  = None
            if btn:
                try:
                    btn.config(text="> Play", bg='#1a6b9e')
                except Exception:
                    pass
    threading.Thread(target=_watch, daemon=True).start()


def play_file(path):
    play_toggle(path, None)


def add_row(parent, label, var, row, filetypes=None, save=False, multi=False, initialdir=None):
    parent.grid_columnconfigure(1, weight=1)
    tk.Label(parent, text=label, anchor='w', width=20).grid(
        row=row, column=0, sticky='w', padx=6, pady=3)
    tk.Entry(parent, textvariable=var).grid(
        row=row, column=1, sticky='ew', padx=4, pady=3)
    col = 2
    if multi:
        tk.Button(parent, text="Browse", width=9,
            command=lambda: browse_files(var, filetypes, initialdir)).grid(
            row=row, column=col, padx=4, pady=3)
    else:
        tk.Button(parent, text="Browse", width=9,
            command=lambda: browse_file(var, filetypes, save, initialdir)).grid(
            row=row, column=col, padx=4, pady=3)


def add_console(parent, start_row):
    parent.grid_rowconfigure(start_row, weight=1)
    parent.grid_columnconfigure(1, weight=1)
    console = scrolledtext.ScrolledText(
        parent, height=18, bg='#1e1e1e', fg='#d4d4d4',
        font=('Courier', 9), state='normal')
    console.grid(row=start_row, column=0, columnspan=3,
                 sticky='nsew', padx=8, pady=4)
    _make_readonly(console)
    return console

def log(console, text):
    if _root:
        _root.after(0, lambda c=console, t=text: (c.config(state='normal'), c.insert('end', t + '\n'), c.see('end')))
    # On laisse state='normal' pour permettre la sélection/copie
    # On bloque juste les touches qui modifient le texte via binding

def _make_readonly(widget):
    """
    Read-only but selectable/copyable console widget.
    Supported shortcuts:
      Ctrl+C       → copy selection
      Ctrl+Insert  → copy selection
      Ctrl+A       → select all
      Ctrl+V       → no effect (read-only)
      Shift+Insert → no effect (read-only)
      Shift+Delete → no effect (read-only)
    """
    def block_edit(e):
        ctrl  = e.state & 0x4
        shift = e.state & 0x1

        # Ctrl+C ou Ctrl+Insert → copier
        if ctrl and e.keysym in ('c', 'C', 'Insert'):
            return
        # Ctrl+A → tout sélectionner
        if ctrl and e.keysym in ('a', 'A'):
            widget.tag_add('sel', '1.0', 'end')
            return 'break'
        # Navigation et sélection clavier
        if e.keysym in ('Left','Right','Up','Down','Home','End',
                        'Prior','Next','Shift_L','Shift_R',
                        'Control_L','Control_R','Alt_L','Alt_R'):
            return
        # Bloquer tout le reste (pas d'édition)
        return 'break'

    widget.bind('<Key>', block_edit)
    widget.config(cursor='xterm')

    # Menu clic droit
    menu = tk.Menu(widget, tearoff=0)
    menu.add_command(label="Copy",             command=lambda: widget.event_generate('<<Copy>>'))
    menu.add_command(label="Select all",       command=lambda: widget.tag_add('sel','1.0','end'))
    menu.add_separator()
    menu.add_command(label="Clear console",    command=lambda: [widget.config(state='normal'),
                                                                 widget.delete('1.0','end')])

    def show_menu(e):
        menu.tk_popup(e.x_root, e.y_root)

    widget.bind('<Button-3>', show_menu)

    # Copie auto dans clipboard X11 sur sélection souris
    def on_select(e):
        try:
            sel = widget.get('sel.first', 'sel.last')
            if sel:
                widget.clipboard_clear()
                widget.clipboard_append(sel)
        except tk.TclError:
            pass
    widget.bind('<ButtonRelease-1>', on_select)

def run_cmd(cmd, console, btn, stop_btn=None, line_callback=None):
    import time as _t, re as _re
    proc_holder = [None]
    timer_on = [False]

    def _setinfo(s):
        if _root and hasattr(btn, '_info_var'):
            _root.after(0, lambda: btn._info_var.set(s))

    def _tick(t0, prog):
        if not timer_on[0]: return
        e = int(_t.time()-t0); h,m,s = e//3600,(e%3600)//60,e%60
        _setinfo(f"[{h:02d}:{m:02d}:{s:02d}]  {prog[0]}")
        if timer_on[0] and _root: _root.after(1000, lambda: _tick(t0, prog))

    def _run():
        btn.config(state='disabled', text='... Running...')
        if stop_btn: stop_btn.config(state='normal')
        log(console, ">  " + " ".join(str(c) for c in cmd) + "\n")
        t0 = _t.time(); prog = [""]
        timer_on[0] = True
        if _root: _root.after(1000, lambda: _tick(t0, prog))
        try:
            env = os.environ.copy()
            env['PYTHONUNBUFFERED'] = '1'
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, env=env,
                start_new_session=False)  # keep in same session for killpg
            proc_holder[0] = proc
            for line in proc.stdout:
                txt = line.rstrip()
                import re as _rf
                txt = _rf.sub(r'<[^>]+_reminder>.*?</[^>]+_reminder>', '', txt, flags=_rf.DOTALL).strip()
                # Hidden progress marker [PROGRESS=n/N] is consumed by the
                # GUI label and NOT echoed to the console.
                m = _re.search(r'\[PROGRESS=(\d+)/(\d+)\]', txt)
                if m:
                    cur, tot = int(m.group(1)), int(m.group(2))
                    pct = int(cur * 100 / tot) if tot else 0
                    prog[0] = f"[{cur}/{tot}] [{pct}%]"
                    # Strip the marker from the line; if nothing else remains,
                    # don't print an empty line.
                    txt = _rf.sub(r'\[PROGRESS=\d+/\d+\]', '', txt).strip()
                if txt:
                    log(console, txt)
                    if line_callback:
                        _root.after(0, lambda t=txt: line_callback(t))
            proc.wait()
            if proc.returncode == 0:   log(console, "\n[OK] Done.")
            elif proc.returncode==-15: log(console, "\n[STOP] Stopped.")
            else:                      log(console, f"\n[ERR] Error (code {proc.returncode})")
        except Exception as e:
            log(console, f"\n[ERR] {e}")
        finally:
            timer_on[0] = False
            # Freeze the final elapsed time + last progress in the info label
            # so it stays visible until the next run.
            final_e = int(_t.time() - t0)
            fh, fm, fs = final_e // 3600, (final_e % 3600) // 60, final_e % 60
            tail = f"  {prog[0]}  done" if prog[0] else "  done"
            _setinfo(f"[{fh:02d}:{fm:02d}:{fs:02d}]{tail}")
            btn.config(state='normal', text=btn._orig_text)
            if stop_btn: stop_btn.config(state='disabled')
            proc_holder[0] = None

    def stop():
        if proc_holder[0]:
            import signal as _sig, os as _os
            try:
                # Kill only the child process and its children, NOT the GUI process
                proc_holder[0].kill()   # SIGKILL directly on child
            except Exception:
                pass
            try:
                # Also kill grandchildren (e.g. whisper spawned by video2txt)
                pgid = _os.getpgid(proc_holder[0].pid)
                gui_pgid = _os.getpgid(_os.getpid())
                if pgid != gui_pgid:   # safety: never kill our own process group
                    _os.killpg(pgid, _sig.SIGKILL)
            except Exception:
                pass
            log(console, "\n[STOP] Stop requested...")

    if stop_btn:
        stop_btn._stop_fn = stop

    threading.Thread(target=_run, daemon=True).start()

def make_btn(parent, text, cmd_fn, row):
    frame = tk.Frame(parent)
    frame.grid(row=row, column=0, columnspan=3, pady=(8,0))
    btn = tk.Button(frame, text=text, command=lambda: cmd_fn(btn, stop_btn),
                    bg='#2d7d46', fg='white', font=('Arial', 10, 'bold'), width=22)
    btn._orig_text = text
    btn.pack(side='left', padx=4)
    stop_btn = tk.Button(frame, text="Stop", state='disabled',
                         command=lambda: stop_btn._stop_fn() if hasattr(stop_btn, '_stop_fn') else None,
                         bg='#c0392b', fg='white', font=('Arial', 10, 'bold'), width=8)
    stop_btn.pack(side='left', padx=4)
    info_var = tk.StringVar(value="")
    tk.Label(frame, textvariable=info_var, font=('Courier', 9), fg='#222222', anchor='w').pack(side='left', padx=8)
    btn._info_var = info_var
    stop_btn._info_var = info_var
    return btn, stop_btn


# ── Tab: Generator ─────────────────────────────────────────────────────────

def tab_generator(nb):
    f = ttk.Frame(nb)
    nb.add(f, text="[Gen] Generator")
    f.grid_columnconfigure(1, weight=1)
    f.grid_rowconfigure(7, weight=1)

    v_script  = tk.StringVar()
    v_output  = tk.StringVar()
    v_ambient = tk.StringVar()
    v_music   = tk.StringVar()

    add_row(f, "Script (.txt)",  v_script,  0, [("Text","*.txt"),("All","*.*")], initialdir=DIR_PROMPTS)
    add_row(f, "Output (wav/mp3)", v_output, 1, [("WAV","*.wav"),("MP3","*.mp3"),("FLAC","*.flac"),("OGG","*.ogg"),("All","*.*")], save=True, initialdir=DIR_OUTPUT)

    # ── Voices — one row per voice, each with multi-ref support ──────────────
    voices_frame = tk.LabelFrame(f, text="Voices", padx=4, pady=2)
    voices_frame.grid(row=2, column=0, columnspan=3, sticky='ew', padx=6, pady=3)
    voices_frame.grid_columnconfigure(1, weight=1)
    voice_rows_gen = []

    btn_add_voice = tk.Button(voices_frame, text="+ Add voice",
                              bg='#2d6a2d', fg='white')
    btn_add_voice.pack(anchor='w', padx=4, pady=2)

    def add_gen_voice_row(num):
        v_num  = tk.IntVar(value=num)
        v_refs = tk.StringVar()
        row_f  = tk.Frame(voices_frame)
        row_f.pack(fill='x', padx=2, pady=1, before=btn_add_voice)
        row_f.grid_columnconfigure(1, weight=1)

        tk.Label(row_f, text=f"Voice [{num}]", width=9, anchor='w').pack(side='left')

        entry = tk.Entry(row_f, textvariable=v_refs)
        entry.pack(side='left', fill='x', expand=True, padx=3)

        def browse_refs(v=v_refs):
            files = filedialog.askopenfilenames(
                filetypes=[("Audio","*.wav *.mp3 *.flac *.ogg"),("All","*.*")],
                initialdir=DIR_VOICES)
            if files:
                existing = v.get().strip()
                new_files = " ".join(files)
                v.set((existing + " " + new_files).strip() if existing else new_files)

        tk.Button(row_f, text="Browse", width=8, command=browse_refs).pack(side='left', padx=2)

        entry_tuple = (v_num, v_refs, row_f)

        def remove(e=entry_tuple):
            e[2].destroy()
            voice_rows_gen.remove(e)

        tk.Button(row_f, text="X", width=2, fg='red', command=remove).pack(side='left', padx=1)
        voice_rows_gen.append(entry_tuple)

    add_gen_voice_row(1)

    def add_gen_voice():
        add_gen_voice_row(len(voice_rows_gen) + 1)

    btn_add_voice.config(command=add_gen_voice)

    add_row(f, "Ambient",       v_ambient, 3, [("Audio","*.wav *.mp3 *.flac *.ogg"),("All","*.*")], initialdir=DIR_AMBIENT)
    add_row(f, "Punctual music (1+)", v_music, 4, [("Audio","*.wav *.mp3 *.flac *.ogg"),("All","*.*")], multi=True, initialdir=DIR_PUNCTUAL)

    # ── MP3 output options ────────────────────────────────────────────────────
    v_gen_mp3_bitrate = tk.StringVar(value='192')
    v_gen_mp3_mode    = tk.StringVar(value='cbr')
    tk.Label(f, text="MP3 bitrate (kbps)", anchor='w', width=20).grid(row=5, column=0, sticky='w', padx=6, pady=3)
    frm_gen_mp3 = tk.Frame(f)
    frm_gen_mp3.grid(row=5, column=1, sticky='w', padx=4)
    ttk.Combobox(frm_gen_mp3, textvariable=v_gen_mp3_bitrate, width=6, state='readonly',
        values=['128','160','192','256','320']).pack(side='left')
    ttk.Combobox(frm_gen_mp3, textvariable=v_gen_mp3_mode, width=5, state='readonly',
        values=['cbr','vbr']).pack(side='left', padx=6)
    tk.Label(frm_gen_mp3, text="(only used if output is .mp3)", fg='grey').pack(side='left')

    # ── Éditeur de prompt ────────────────────────────────────────────────────
    editor_frame = ttk.LabelFrame(f, text="Prompt Editor")
    editor_frame.grid(row=7, column=0, columnspan=3, sticky='nsew', padx=6, pady=4)
    editor_frame.grid_columnconfigure(0, weight=1)
    editor_frame.grid_rowconfigure(1, weight=1)

    # Editor toolbar
    btn_bar = tk.Frame(editor_frame)
    btn_bar.grid(row=0, column=0, sticky='ew', padx=4, pady=2)

    def nouveau_prompt():
        editor.delete('1.0', 'end')
        v_script.set('')

    def ouvrir_prompt():
        path = filedialog.askopenfilename(filetypes=[('Text','*.txt'),('All','*.*')])
        if path:
            v_script.set(path)
            with open(path, encoding='utf-8') as fh:
                editor.delete('1.0', 'end')
                editor.insert('1.0', fh.read())

    def sauvegarder_prompt():
        path = v_script.get()
        if not path:
            path = filedialog.asksaveasfilename(
                filetypes=[('Text','*.txt')], defaultextension='.txt')
            if not path: return
            v_script.set(path)
        with open(path, 'w', encoding='utf-8') as fh:
            fh.write(editor.get('1.0', 'end-1c'))
        log_editor(f"Saved: {path}")

    def sauvegarder_sous():
        path = filedialog.asksaveasfilename(
            filetypes=[('Text','*.txt')], defaultextension='.txt')
        if path:
            v_script.set(path)
            with open(path, 'w', encoding='utf-8') as fh:
                fh.write(editor.get('1.0', 'end-1c'))
            log_editor(f"Saved as: {path}")

    tk.Button(btn_bar, text="New",       command=nouveau_prompt,    width=8).pack(side='left', padx=2)
    tk.Button(btn_bar, text="Open",      command=ouvrir_prompt,     width=8).pack(side='left', padx=2)
    tk.Button(btn_bar, text="Save",      command=sauvegarder_prompt,width=8).pack(side='left', padx=2)
    tk.Button(btn_bar, text="Save as",   command=sauvegarder_sous,  width=8).pack(side='left', padx=2)
    ttk.Separator(btn_bar, orient='vertical').pack(side='left', fill='y', padx=6, pady=2)
    tk.Button(btn_bar, text="Find/Replace (Ctrl+H)", width=20,
              command=lambda: open_find_replace()).pack(side='left', padx=2)
    tk.Button(btn_bar, text="Go to line (Ctrl+G)", width=18,
              command=lambda: open_goto_line()).pack(side='left', padx=2)

    # ── Editor area with line numbers ─────────────────────────────────────
    editor_container = tk.Frame(editor_frame)
    editor_container.grid(row=1, column=0, sticky='nsew', padx=4, pady=2)
    editor_container.grid_columnconfigure(1, weight=1)
    editor_container.grid_rowconfigure(0, weight=1)

    line_nums = tk.Text(editor_container, width=4, font=('Courier', 9),
                        bg='#e8e8e8', fg='#888', state='disabled',
                        cursor='arrow', takefocus=False)
    line_nums.grid(row=0, column=0, sticky='ns')

    editor = tk.Text(editor_container, height=12, font=('Courier', 9),
                     wrap='word', undo=True)
    editor.grid(row=0, column=1, sticky='nsew')

    scroll_e = ttk.Scrollbar(editor_frame, orient='vertical')
    scroll_e.grid(row=1, column=1, sticky='ns')

    def sync_scroll(*args):
        editor.yview(*args)
        line_nums.yview(*args)

    scroll_e.config(command=sync_scroll)

    def _on_editor_yscroll(*args):
        scroll_e.set(*args)
        line_nums.yview_moveto(args[0])

    editor.config(yscrollcommand=_on_editor_yscroll)
    line_nums.config(yscrollcommand=scroll_e.set)

    def update_line_numbers(event=None):
        line_nums.config(state='normal')
        line_nums.delete('1.0', 'end')
        n_lines = int(editor.index('end-1c').split('.')[0])
        line_nums.insert('1.0', '\n'.join(str(i) for i in range(1, n_lines + 1)))
        line_nums.config(state='disabled')

    editor.bind('<KeyRelease>', update_line_numbers)
    editor.bind('<Button-1>',   lambda e: editor.focus_set())
    editor.bind('<Button-2>',   lambda e: editor.focus_set())  # X11 middle-click

    # Sync mousewheel scroll between editor and line numbers
    def _on_mousewheel(event):
        editor.yview_scroll(int(-1 * (event.delta / 120)), "units")
        line_nums.yview_scroll(int(-1 * (event.delta / 120)), "units")
        return "break"

    def _on_scroll_up(event):   # Linux Button-4
        editor.yview_scroll(-1, "units")
        line_nums.yview_scroll(-1, "units")
        return "break"

    def _on_scroll_down(event):  # Linux Button-5
        editor.yview_scroll(1, "units")
        line_nums.yview_scroll(1, "units")
        return "break"

    editor.bind("<MouseWheel>", _on_mousewheel)   # Windows/macOS
    editor.bind("<Button-4>",   _on_scroll_up)    # Linux scroll up
    editor.bind("<Button-5>",   _on_scroll_down)  # Linux scroll down

    # Also block line_nums from scrolling independently
    line_nums.bind("<MouseWheel>", _on_mousewheel)
    line_nums.bind("<Button-4>",   _on_scroll_up)
    line_nums.bind("<Button-5>",   _on_scroll_down)

    # ── Right-click context menu ──────────────────────────────────────────
    ctx_menu = tk.Menu(editor, tearoff=0)
    ctx_menu.add_command(label="Cut",        command=lambda: editor.event_generate('<<Cut>>'))
    ctx_menu.add_command(label="Copy",       command=lambda: editor.event_generate('<<Copy>>'))
    ctx_menu.add_command(label="Paste",      command=lambda: editor.event_generate('<<Paste>>'))
    ctx_menu.add_separator()
    ctx_menu.add_command(label="Select All", command=lambda: editor.tag_add('sel','1.0','end'))
    ctx_menu.add_separator()
    ctx_menu.add_command(label="Find/Replace...", command=lambda: open_find_replace())
    ctx_menu.add_command(label="Go to line...",   command=lambda: open_goto_line())

    def show_ctx_menu(event):
        try:
            ctx_menu.tk_popup(event.x_root, event.y_root)
        finally:
            ctx_menu.grab_release()

    editor.bind('<Button-3>', show_ctx_menu)

    # ── Keyboard shortcuts ────────────────────────────────────────────────
    editor.bind('<Control-h>', lambda e: (open_find_replace(), 'break'))
    editor.bind('<Control-H>', lambda e: (open_find_replace(), 'break'))
    editor.bind('<Control-g>', lambda e: (open_goto_line(),    'break'))
    editor.bind('<Control-G>', lambda e: (open_goto_line(),    'break'))
    editor.bind('<Control-s>', lambda e: (sauvegarder_prompt(),'break'))
    editor.bind('<Control-S>', lambda e: (sauvegarder_prompt(),'break'))

    # ── Find & Replace dialog ─────────────────────────────────────────────
    _fr_win = [None]

    def open_find_replace():
        if _fr_win[0] and _fr_win[0].winfo_exists():
            _fr_win[0].lift(); return

        win = tk.Toplevel(editor_frame)
        win.title("Find & Replace")
        win.resizable(False, False)
        win.transient(editor_frame)
        _fr_win[0] = win

        tk.Label(win, text="Find:",    width=8, anchor='e').grid(row=0, column=0, padx=4, pady=4)
        tk.Label(win, text="Replace:", width=8, anchor='e').grid(row=1, column=0, padx=4, pady=4)

        v_find    = tk.StringVar()
        v_replace = tk.StringVar()
        e_find    = tk.Entry(win, textvariable=v_find,    width=32)
        e_replace = tk.Entry(win, textvariable=v_replace, width=32)
        e_find.grid(   row=0, column=1, columnspan=2, padx=4, pady=4, sticky='ew')
        e_replace.grid(row=1, column=1, columnspan=2, padx=4, pady=4, sticky='ew')
        e_find.focus_set()

        v_case = tk.BooleanVar(value=False)
        tk.Checkbutton(win, text="Match case", variable=v_case).grid(
            row=2, column=1, sticky='w', padx=4)

        status_fr = tk.StringVar(value="")
        tk.Label(win, textvariable=status_fr, fg='gray', width=30, anchor='w').grid(
            row=3, column=1, columnspan=2, padx=4)

        def find_next(start='insert'):
            editor.tag_remove('found', '1.0', 'end')
            needle = v_find.get()
            if not needle: return
            nocase = not v_case.get()
            pos = editor.search(needle, start, stopindex='end', nocase=nocase)
            if pos:
                end = f"{pos}+{len(needle)}c"
                editor.tag_add('found', pos, end)
                editor.tag_config('found', background='#ffff00', foreground='#000')
                editor.mark_set('insert', end)
                editor.see(pos)
                status_fr.set(f"Found at line {pos.split('.')[0]}")
            else:
                status_fr.set("Not found — wrapping...")
                pos2 = editor.search(needle, '1.0', stopindex='end', nocase=nocase)
                if pos2:
                    end2 = f"{pos2}+{len(needle)}c"
                    editor.tag_add('found', pos2, end2)
                    editor.tag_config('found', background='#ffff00', foreground='#000')
                    editor.mark_set('insert', end2)
                    editor.see(pos2)
                    status_fr.set(f"Wrapped — found at line {pos2.split('.')[0]}")
                else:
                    status_fr.set("Not found.")

        def replace_one():
            needle  = v_find.get()
            rep     = v_replace.get()
            nocase  = not v_case.get()
            if editor.tag_ranges('found'):
                start, end = str(editor.tag_ranges('found')[0]), str(editor.tag_ranges('found')[1])
                editor.delete(start, end)
                editor.insert(start, rep)
                editor.tag_remove('found', '1.0', 'end')
                find_next(start)
            else:
                find_next()

        def replace_all():
            needle = v_find.get()
            rep    = v_replace.get()
            nocase = not v_case.get()
            if not needle: return
            count = 0
            pos = '1.0'
            while True:
                pos = editor.search(needle, pos, stopindex='end', nocase=nocase)
                if not pos: break
                end = f"{pos}+{len(needle)}c"
                editor.delete(pos, end)
                editor.insert(pos, rep)
                pos = f"{pos}+{len(rep)}c"
                count += 1
            status_fr.set(f"Replaced {count} occurrence(s).")

        btn_frame = tk.Frame(win)
        btn_frame.grid(row=4, column=0, columnspan=3, pady=6)
        tk.Button(btn_frame, text="Find Next",    width=12, command=find_next).pack(side='left', padx=4)
        tk.Button(btn_frame, text="Replace",      width=12, command=replace_one).pack(side='left', padx=4)
        tk.Button(btn_frame, text="Replace All",  width=12, command=replace_all).pack(side='left', padx=4)
        tk.Button(btn_frame, text="Close",        width=8,  command=win.destroy).pack(side='left', padx=4)

        e_find.bind('<Return>',   lambda e: find_next())
        e_find.bind('<KP_Enter>', lambda e: find_next())
        win.bind('<Escape>',      lambda e: win.destroy())

    # ── Go to line dialog ─────────────────────────────────────────────────
    def open_goto_line():
        win = tk.Toplevel(editor_frame)
        win.title("Go to line")
        win.resizable(False, False)
        win.transient(editor_frame)

        n_lines = int(editor.index('end-1c').split('.')[0])
        tk.Label(win, text=f"Line (1–{n_lines}):").grid(row=0, column=0, padx=8, pady=8)
        v_line = tk.StringVar()
        e_line = tk.Entry(win, textvariable=v_line, width=8)
        e_line.grid(row=0, column=1, padx=4)
        e_line.focus_set()

        def go():
            try:
                n = int(v_line.get())
                n = max(1, min(n, n_lines))
                editor.mark_set('insert', f'{n}.0')
                editor.see(f'{n}.0')
                editor.focus_set()
                win.destroy()
            except ValueError:
                pass

        tk.Button(win, text="Go", command=go, width=6).grid(row=0, column=2, padx=4)
        e_line.bind('<Return>',   lambda e: go())
        e_line.bind('<KP_Enter>', lambda e: go())
        win.bind('<Escape>',      lambda e: win.destroy())

    # ── Status bar ────────────────────────────────────────────────────────
    status_var = tk.StringVar(value="Ready  |  Ctrl+S: Save  |  Ctrl+H: Find/Replace  |  Ctrl+G: Go to line  |  Right-click: menu")
    status_lbl = tk.Label(editor_frame, textvariable=status_var,
                          anchor='w', font=('Arial', 8), fg='gray')
    status_lbl.grid(row=2, column=0, columnspan=2, sticky='ew', padx=4)

    def update_cursor_pos(event=None):
        pos  = editor.index('insert')
        line, col = pos.split('.')
        status_var.set(f"Line {line}, Col {int(col)+1}  |  Ctrl+S: Save  |  Ctrl+H: Find/Replace  |  Ctrl+G: Go to line")
        update_line_numbers()

    editor.bind('<KeyRelease>',   update_cursor_pos)
    editor.bind('<ButtonRelease>', update_cursor_pos)

    def log_editor(msg):
        status_var.set(msg)

    def on_script_change(*args):
        path = v_script.get()
        if path and os.path.exists(path):
            try:
                with open(path, encoding='utf-8') as fh:
                    content = fh.read()
                if editor.get('1.0', 'end-1c') != content:
                    editor.delete('1.0', 'end')
                    editor.insert('1.0', content)
                    log_editor(f"Loaded: {os.path.basename(path)}")
            except Exception:
                pass

    v_script.trace_add('write', on_script_change)

    # Console output
    console_frame = ttk.LabelFrame(f, text="Console")
    console_frame.grid(row=8, column=0, columnspan=3, sticky='ew', padx=6, pady=4)
    console_frame.grid_columnconfigure(0, weight=1)
    console = scrolledtext.ScrolledText(console_frame, height=10, bg='#1e1e1e',
                                         fg='#d4d4d4', font=('Courier', 9), state='normal')
    console.grid(row=0, column=0, sticky='ew', padx=4, pady=4)
    _make_readonly(console)

    def lancer(btn, stop_btn=None):
        # Sauvegarder automatiquement avant de lancer
        if editor.get('1.0', 'end-1c').strip():
            sauvegarder_prompt()

        # Collect voices — separate rows with "--" so generator knows voice boundaries
        voice_args = []
        valid_rows = [(vn, vr, vf) for vn, vr, vf in voice_rows_gen if vr.get().strip()]
        for i, (_vnum, _vrefs, _vframe) in enumerate(valid_rows):
            refs = _vrefs.get().strip()
            files = [rf.strip() for rf in refs.split() if rf.strip() and rf.strip() != '+']
            voice_args += files
            if i < len(valid_rows) - 1:
                voice_args.append('--')  # separator between voices

        if not v_script.get() or not v_output.get() or not voice_args:
            log(console, "[ERR] Script, output and at least one voice required."); return
        cmd = [sys.executable,
               os.path.join(SCRIPTS_DIR, 'guided_meditation_generator_v23.py'),
               v_script.get(), v_output.get()]
        cmd += voice_args
        if v_ambient.get(): cmd += v_ambient.get().split()
        if v_music.get():   cmd += v_music.get().split()
        cmd += ['--mp3-bitrate', v_gen_mp3_bitrate.get(), '--mp3-mode', v_gen_mp3_mode.get()]
        run_cmd(cmd, console, btn, stop_btn)

    make_btn(f, ">  Run", lancer, 9)


LANGS = ['FR','EN','ES','DE','IT','PT','PL','TR','RU','NL','CS','AR','ZH-CN','HU','KO','JA','HI']

# ── Tab: Analyser ───────────────────────────────────────────────────────────

# ── Tab: Auto pipeline ────────────────────────────────────────────────────────

def tab_auto(nb):
    f = ttk.Frame(nb)
    nb.add(f, text="[Auto] Pipeline")
    f.grid_columnconfigure(0, weight=1)

    voices_frame = tk.LabelFrame(f, text="Voices (reference + language)")
    voices_frame.grid(row=0, column=0, columnspan=3, sticky='ew', padx=6, pady=4)
    auto_rows = []

    def add_auto_row():
        v_path = tk.StringVar()
        v_lang = tk.StringVar(value='FR')
        row_f = tk.Frame(voices_frame)
        row_f.pack(fill='x', padx=4, pady=2)
        tk.Label(row_f, text=f"V{len(auto_rows)+1}", width=3).pack(side='left')
        tk.Entry(row_f, textvariable=v_path).pack(side='left', fill='x', expand=True, padx=3)
        def browse():
            p = filedialog.askopenfilename(
                filetypes=[("Audio", "*.wav *.mp3 *.flac *.ogg"), ("All", "*.*")],
                initialdir=DIR_VOICES)
            if p:
                v_path.set(p)
        tk.Button(row_f, text="...", command=browse).pack(side='left', padx=1)
        ttk.Combobox(row_f, textvariable=v_lang, values=LANGS, width=7,
                     state='readonly').pack(side='left', padx=2)
        def remove():
            row_f.destroy(); auto_rows.remove(entry)
        tk.Button(row_f, text="✕", command=remove).pack(side='left', padx=1)
        entry = (v_path, v_lang, row_f)
        auto_rows.append(entry)

    add_auto_row()
    tk.Button(f, text="+ Add voice", command=add_auto_row).grid(row=1, column=0,
                                                                sticky='w', padx=8, pady=2)

    v_a_seeds  = tk.StringVar(value='0 42 100 180 200')
    v_a_budget = tk.StringVar(value='60')
    v_a_keep   = tk.StringVar(value='45')
    v_a_wacc   = tk.StringVar(value='0.6')
    v_a_wid    = tk.StringVar(value='0.4')
    v_a_curate = tk.BooleanVar(value=True)
    v_a_autotx = tk.BooleanVar(value=True)
    v_a_beams  = tk.BooleanVar(value=False)
    v_a_screen = tk.BooleanVar(value=False)
    v_a_target = tk.StringVar(value='')          # empty = match the reference
    v_a_optaud = tk.StringVar(value='none')
    try:
        import torch as _t_auto
        _auto_dev = 'cuda' if _t_auto.cuda.is_available() else 'cpu'
    except Exception:
        _auto_dev = 'cpu'
    v_a_device = tk.StringVar(value=_auto_dev)

    frm_a = tk.Frame(f); frm_a.grid(row=2, column=0, columnspan=3, sticky='w', padx=6, pady=3)
    tk.Label(frm_a, text="Seeds").pack(side='left', padx=(6, 2))
    tk.Entry(frm_a, textvariable=v_a_seeds, width=20).pack(side='left')
    for lbl, var, w in [("Budget", v_a_budget, 4), ("Keep s", v_a_keep, 4),
                        ("w_acc", v_a_wacc, 4), ("w_id", v_a_wid, 4)]:
        tk.Label(frm_a, text=lbl).pack(side='left', padx=(8, 2))
        tk.Entry(frm_a, textvariable=var, width=w).pack(side='left')
    tk.Label(frm_a, text="Device").pack(side='left', padx=(8, 2))
    ttk.Combobox(frm_a, textvariable=v_a_device, values=['cpu', 'cuda'],
                 width=6, state='readonly').pack(side='left')

    frm_b = tk.Frame(f); frm_b.grid(row=3, column=0, columnspan=3, sticky='w', padx=6)
    tk.Checkbutton(frm_b, text="Curate reference", variable=v_a_curate).pack(side='left', padx=6)
    tk.Checkbutton(frm_b, text="Fit on reference's own words (auto-text)",
                   variable=v_a_autotx).pack(side='left', padx=6)
    tk.Checkbutton(frm_b, text="Probe beam/greedy decoding",
                   variable=v_a_beams).pack(side='left', padx=6)

    frm_c = tk.Frame(f); frm_c.grid(row=4, column=0, columnspan=3, sticky='w', padx=6, pady=2)
    tk.Checkbutton(frm_c, text="Screen audio params (which knobs move identity)",
                   variable=v_a_screen).pack(side='left', padx=6)
    tk.Label(frm_c, text="Optimise audio").pack(side='left', padx=(10, 2))
    ttk.Combobox(frm_c, textvariable=v_a_optaud, values=['none', 'nelder', 'de'],
                 width=7, state='readonly').pack(side='left')
    tk.Label(frm_c, text="Target dBFS").pack(side='left', padx=(10, 2))
    tk.Entry(frm_c, textvariable=v_a_target, width=6).pack(side='left')
    tk.Label(frm_c, text="(empty = match reference; -20 = production level)",
             fg='grey', font=("Arial", 8)).pack(side='left', padx=(4, 0))

    tk.Label(f, text="Runs curate → analyse → optimise → tone-fit for each voice and"
                     " prints the final {} / [] blocks to paste. LISTEN to each"
                     " *_pipeline_clone.wav before generating — scores don't hear"
                     " naturalness.",
             fg='gray', font=("Arial", 8), justify='left', wraplength=560,
             anchor='w').grid(row=5, column=0, columnspan=3, sticky='w', padx=8)

    console = add_console(f, 7)

    def lancer(btn, stop_btn=None):
        voices = [(v.get().strip(), l.get()) for v, l, _ in auto_rows if v.get().strip()]
        if not voices:
            log(console, "[ERR] At least one voice required."); return
        cmd = [sys.executable, os.path.join(SCRIPTS_DIR, 'xtts_pipeline.py')]
        for path, lg in voices:
            cmd += ['--voice', path, lg]
        cmd += ['--seeds', v_a_seeds.get().strip() or '0 42 100 180 200']
        cmd += ['--budget', v_a_budget.get().strip() or '60']
        cmd += ['--keep-seconds', v_a_keep.get().strip() or '45']
        cmd += ['--w-accent', v_a_wacc.get().strip() or '0.6']
        cmd += ['--w-identity', v_a_wid.get().strip() or '0.4']
        cmd += ['--device', v_a_device.get()]
        if not v_a_curate.get():
            cmd += ['--no-curate']
        if not v_a_autotx.get():
            cmd += ['--no-auto-text']
        if v_a_beams.get():
            cmd += ['--probe-beams']
        if v_a_screen.get():
            cmd += ['--screen-audio']
        if v_a_optaud.get() != 'none':
            cmd += ['--optimise-audio', v_a_optaud.get()]
        if v_a_target.get().strip():
            cmd += ['--target-dbfs', v_a_target.get().strip()]
        run_cmd(cmd, console, btn, stop_btn)

    make_btn(f, ">  Run full pipeline", lancer, 6)


# ── Tab: Curation ─────────────────────────────────────────────────────────────

def tab_curate(nb):
    f = ttk.Frame(nb)
    nb.add(f, text="[Cur] Curation")
    f.grid_columnconfigure(1, weight=1)

    v_cur_input   = tk.StringVar()
    v_cur_output  = tk.StringVar()
    v_cur_keep    = tk.StringVar(value='45')
    v_cur_window  = tk.StringVar(value='4')
    v_cur_hop     = tk.StringVar(value='2')
    v_cur_minsc   = tk.StringVar(value='auto')
    try:
        import torch as _t_cur
        _cur_dev = 'cuda' if _t_cur.cuda.is_available() else 'cpu'
    except Exception:
        _cur_dev = 'cpu'
    v_cur_device  = tk.StringVar(value=_cur_dev)

    add_row(f, "Voice ref(s)", v_cur_input, 0,
            [("Audio", "*.wav *.mp3 *.flac *.ogg"), ("All", "*.*")],
            multi=True, initialdir=DIR_VOICES)
    add_row(f, "Curated output", v_cur_output, 1,
            [("WAV", "*.wav")], save=True, initialdir=DIR_VOICES)

    def _suggest_out(*_):
        # Auto-fill "<first_ref>_curated.wav" when output is empty
        if v_cur_input.get().strip() and not v_cur_output.get().strip():
            first = v_cur_input.get().split()[0]
            base, _ext = os.path.splitext(first)
            v_cur_output.set(base + "_curated.wav")
    v_cur_input.trace_add('write', _suggest_out)

    frm_c = tk.Frame(f); frm_c.grid(row=2, column=0, columnspan=3, sticky='w', padx=6, pady=3)
    for lbl, var, w in [("Keep seconds", v_cur_keep, 5), ("Window s", v_cur_window, 4),
                        ("Hop s", v_cur_hop, 4), ("Min score", v_cur_minsc, 6)]:
        tk.Label(frm_c, text=lbl).pack(side='left', padx=(6, 2))
        tk.Entry(frm_c, textvariable=var, width=w).pack(side='left')
    tk.Label(frm_c, text="Device").pack(side='left', padx=(10, 2))
    ttk.Combobox(frm_c, textvariable=v_cur_device, values=['cpu', 'cuda'],
                 width=6, state='readonly').pack(side='left')

    tk.Label(f, text="Keeps only the most speaker-coherent windows (ECAPA) — breaths,"
                     " noise, off-voice segments are dropped. Run everything downstream"
                     " (Analyser/Optimiser/Comparator) on the curated file.",
             fg='gray', font=("Arial", 8), justify='left', wraplength=560,
             anchor='w').grid(row=3, column=0, columnspan=3, sticky='w', padx=8)

    console = add_console(f, 5)

    def lancer(btn, stop_btn=None):
        refs = [r for r in v_cur_input.get().split() if r.strip()]
        if not refs:
            log(console, "[ERR] At least one voice reference required."); return
        if not v_cur_output.get().strip():
            log(console, "[ERR] Output file required."); return
        cmd = [sys.executable, os.path.join(SCRIPTS_DIR, 'curate_reference.py')]
        cmd += refs
        cmd += ['-o', v_cur_output.get().strip()]
        cmd += ['--keep-seconds', v_cur_keep.get().strip() or '45']
        cmd += ['--window', v_cur_window.get().strip() or '4']
        cmd += ['--hop', v_cur_hop.get().strip() or '2']
        cmd += ['--min-score', v_cur_minsc.get().strip() or 'auto']
        cmd += ['--device', v_cur_device.get()]
        run_cmd(cmd, console, btn, stop_btn)

    make_btn(f, ">  Curate", lancer, 4)

    frm_cho = tk.Frame(f)
    frm_cho.grid(row=6, column=0, columnspan=3, sticky='w', padx=6, pady=(0, 4))
    tk.Label(frm_cho, text="Use curated file →", fg="gray",
             font=("Arial", 8)).pack(side='left', padx=(0, 4))
    def _cur_to(target, name):
        out = v_cur_output.get().strip()
        if not out or not os.path.exists(out):
            log(console, "[!] Run the curation first."); return
        ok = _handoff_set(target, out)
        log(console, f"[→] Sent curated file to {name}" if ok else f"[!] {name} tab not ready.")
    tk.Button(frm_cho, text="→ Analyser",
              command=lambda: _cur_to('ana_voice1', 'Analyser')).pack(side='left', padx=2)
    tk.Button(frm_cho, text="→ Optimiser",
              command=lambda: _cur_to('opt_voices', 'Optimiser')).pack(side='left', padx=2)


def tab_analyser(nb):
    f = ttk.Frame(nb)
    nb.add(f, text="[Ana] Analyser")
    f.grid_columnconfigure(0, weight=1)
    f.grid_rowconfigure(2, weight=1)

    voice_rows = []

    # Voice list frame
    voices_frame = tk.LabelFrame(f, text="Voices to analyse")
    voices_frame.grid(row=0, column=0, columnspan=3, sticky='ew', padx=6, pady=4)

    def add_voice_row(num=None):
        if num is None:
            num = len(voice_rows) + 1
        v_path    = tk.StringVar()
        v_lang    = tk.StringVar(value='FR')
        v_num     = tk.IntVar(value=num)
        v_seed    = tk.IntVar(value=0)
        v_precise  = tk.BooleanVar(value=False)
        v_f0       = tk.StringVar(value='auto')
        v_analysis = tk.StringVar(value='Praat')

        row_f = tk.Frame(voices_frame)
        row_f.pack(fill='x', padx=4, pady=2)

        tk.Label(row_f, text="V", width=2).pack(side='left')
        tk.Spinbox(row_f, from_=1, to=20, textvariable=v_num, width=3).pack(side='left', padx=1)
        tk.Entry(row_f, textvariable=v_path).pack(side='left', fill='x', expand=True, padx=3)

        def browse_multi_refs(vp=v_path):
            files = filedialog.askopenfilenames(
                filetypes=[("Audio","*.wav *.mp3 *.flac *.ogg"),("All","*.*")],
                initialdir=DIR_VOICES)
            if files:
                existing = vp.get().strip()
                new_files = " ".join(files)
                vp.set((existing + " " + new_files).strip() if existing else new_files)

        tk.Button(row_f, text="Browse", width=8,
            command=browse_multi_refs).pack(side='left', padx=1)

        ttk.Combobox(row_f, textvariable=v_lang, values=LANGS,
                     width=6, state='readonly').pack(side='left', padx=1)
        tk.Label(row_f, text="Seed:").pack(side='left', padx=(4,0))
        tk.Spinbox(row_f, from_=0, to=99999, textvariable=v_seed, width=6).pack(side='left', padx=1)

        tk.Checkbutton(row_f, text="Prec", variable=v_precise,
                       command=lambda: _on_prec_toggle()).pack(side='left', padx=1)

        cb_f0 = ttk.Combobox(row_f, textvariable=v_f0, values=['auto','crepe','pyin'],
                              width=6, state='readonly')
        lbl_none = tk.Label(row_f, text='none', relief='sunken',
                            bg='#d9d9d9', fg='#888')

        cb_analysis = ttk.Combobox(row_f, textvariable=v_analysis,
                                   values=['Praat', 'Librosa'],
                                   width=7, state='readonly')
        cb_analysis.pack(side='left', padx=2)

        entry = (v_path, v_lang, v_num, v_seed, v_precise, row_f, v_f0, v_analysis)

        def remove(e=entry):
            e[5].destroy()
            voice_rows.remove(e)

        btn_x = tk.Button(row_f, text="X", width=2, fg='red', command=remove)

        lbl_none.pack(side='left', padx=2, ipadx=16)  # shown by default
        btn_x.pack(side='left', padx=1)

        def _on_prec_toggle(var=v_precise, cb=cb_f0, lbl=lbl_none, bx=btn_x):
            if var.get():
                lbl.pack_forget()
                v_f0.set('auto')
                cb.pack(side='left', padx=2, before=bx)
            else:
                cb.pack_forget()
                lbl.pack(side='left', padx=2, ipadx=16, before=bx)

        voice_rows.append(entry)

    add_voice_row(1)
    add_voice_row(2)
    TARGETS['ana_voice1'] = voice_rows[0][0]   # Curation -> Analyser hand-off

    # Add button + precise option
    ctrl_frame = tk.Frame(f)
    ctrl_frame.grid(row=1, column=0, columnspan=3, sticky='w', padx=6, pady=2)
    tk.Button(ctrl_frame, text="+ Add voice",
        command=lambda: add_voice_row(),
        bg='#444', fg='white', width=14).pack(side='left', padx=4)
    tk.Label(ctrl_frame, text="Prec = precise mode  |  F0: auto/crepe/pyin  |  Analysis: Praat / Librosa",
             fg='gray', font=('Arial',8)).pack(side='left', padx=6)

    console = add_console(f, 2)

    def lancer(btn, stop_btn=None):
        valids = [(vp.get(), vl.get(), vn.get(), vs.get(), vpr.get(), vf0.get(), van.get())
                  for vp, vl, vn, vs, vpr, _, vf0, van in voice_rows if vp.get().strip()]
        if not valids:
            log(console, "[ERR] Add at least one voice."); return

        cmds = []
        for vpath, vlang, vnum, vseed, vprec, vf0eng, vanalysis in valids:
            cmd = [sys.executable, os.path.join(SCRIPTS_DIR, 'voice_analyser.py')]
            if vprec:
                cmd.append('--precise')
                cmd += ['--f0-engine', vf0eng]
            if vanalysis == 'Librosa':
                cmd.append('--no-praat')
            cmd += ['--start-num', str(vnum)]
            if vseed != 0:
                cmd += ['--seed', str(vseed)]
            # support multiple space-separated reference files per voice
            ref_files = [f for f in vpath.split() if f.strip()]
            cmd += ref_files
            cmd += [vlang]
            cmds.append(cmd)

        # Run commands sequentially in a thread
        import subprocess, threading
        proc_holder = [None]

        def _run_all():
            import time as _t2
            btn.config(state='disabled', text='... Running')
            if stop_btn: stop_btn.config(state='normal')
            t0 = _t2.time(); ta = [True]
            def _tick2():
                if not ta[0]: return
                e=int(_t2.time()-t0); h,m,s=e//3600,(e%3600)//60,e%60
                if _root and hasattr(btn,'_info_var'):
                    _root.after(0, lambda: btn._info_var.set(f"[{h:02d}:{m:02d}:{s:02d}]"))
                if ta[0] and _root: _root.after(1000, _tick2)
            if _root: _root.after(1000, _tick2)
            summary_lines = []
            try:
                for cmd in cmds:
                    log(console, "\n> " + " ".join(str(c) for c in cmd))
                    proc = subprocess.Popen(
                        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        text=True, bufsize=1)
                    proc_holder[0] = proc
                    voice_lines = []
                    for line in proc.stdout:
                        log(console, line.rstrip())
                        voice_lines.append(line.rstrip())
                    proc.wait()
                    if proc.returncode != 0:
                        log(console, f"[ERR] code {proc.returncode}")
                        break
                    # Extract bracket blocks from output
                    block = []
                    for line in voice_lines:
                        s = line.strip()
                        if s.startswith('# Voice') or s.startswith('# voice'):
                            block = [s]
                        elif block and (s.startswith('{') or s.startswith('[')):
                            block.append(s)
                            if s.startswith('{'):
                                HANDOFF['xtts'] = s
                            elif s.startswith('[1'):
                                HANDOFF['audio'] = s
                    if block:
                        summary_lines.append('\n'.join(block))

                # Show final summary if multiple voices
                if len(summary_lines) > 1:
                    log(console, "\n" + "="*62)
                    log(console, "  FINAL SUMMARY — ready to paste")
                    log(console, "="*62)
                    for block in summary_lines:
                        log(console, "\n" + block)
                    log(console, "")

                log(console, "\n[OK] Analysis complete.")
            except Exception as e:
                log(console, f"\n[ERR] {e}")
            finally:
                ta[0] = False
                # Freeze final elapsed time in the info label so it stays
                # visible until the next run.
                final_e = int(_t2.time() - t0)
                fh, fm, fs = final_e // 3600, (final_e % 3600) // 60, final_e % 60
                if _root and hasattr(btn, '_info_var'):
                    _root.after(0, lambda: btn._info_var.set(
                        f"[{fh:02d}:{fm:02d}:{fs:02d}] done"))
                btn.config(state='normal', text='> Analyse')
                if stop_btn: stop_btn.config(state='disabled')
                proc_holder[0] = None

        if stop_btn:
            def stop():
                if proc_holder[0]:
                    proc_holder[0].terminate()
                    log(console, "\nStop demandé...")
            stop_btn._stop_fn = stop

        threading.Thread(target=_run_all, daemon=True).start()

    make_btn(f, "> Analyse", lancer, 3)

    # Hand-off: push the analysed {} / [] into the Validator or Comparator fields
    frm_ho = tk.Frame(f)
    frm_ho.grid(row=4, column=0, columnspan=2, sticky='w', padx=6, pady=(0, 4))
    tk.Label(frm_ho, text="Send result →", fg="gray", font=("Arial", 8)).pack(side='left', padx=(0, 4))
    def _to_validator():
        ok = _handoff_set('val_xtts', HANDOFF['xtts']) | _handoff_set('val_audio', HANDOFF['audio'])
        log(console, "[→] Sent {}/[] to Validator" if ok else "[!] Run the analysis first.")
    def _to_comparator():
        ok = _handoff_set('cmp_xtts', HANDOFF['xtts']) | _handoff_set('cmp_audio', HANDOFF['audio'])
        log(console, "[→] Sent {}/[] to Comparator" if ok else "[!] Run the analysis first.")
    tk.Button(frm_ho, text="→ Validator", command=_to_validator).pack(side='left', padx=2)
    def _to_optimize():
        ok = _handoff_set('opt_xtts', HANDOFF['xtts'])
        log(console, "[→] Sent {} to Optimiser" if ok else "[!] Run the analysis first.")
    tk.Button(frm_ho, text="→ Optimise", command=_to_optimize).pack(side='left', padx=2)
    tk.Button(frm_ho, text="→ Comparator", command=_to_comparator).pack(side='left', padx=2)


# ── Tab: Transcription ──────────────────────────────────────────────────────

def tab_transcribe(nb):
    f = ttk.Frame(nb)
    nb.add(f, text="[Txt] Transcription")
    f.grid_columnconfigure(1, weight=1)

    v_input  = tk.StringVar()
    v_output = tk.StringVar()
    v_model  = tk.StringVar(value='medium')
    v_pause  = tk.StringVar(value='0.7')
    v_lang   = tk.StringVar(value='fr')
    v_pitch  = tk.BooleanVar(value=False)

    add_row(f, "Video/Audio", v_input,  0,
            [("Video","*.mp4 *.mkv *.avi *.mov *.flv *.webm *.wmv *.m4v *.ts *.mpg"),
             ("Audio","*.mp3 *.wav *.flac *.ogg"),
             ("All","*.*")], initialdir=DIR_MP3)
    add_row(f, "Output (.txt)", v_output, 1, [("Text","*.txt")], save=True, initialdir=DIR_TXT)

    tk.Label(f, text="Whisper model", anchor='w', width=20).grid(row=2, column=0, sticky='w', padx=6, pady=3)
    ttk.Combobox(f, textvariable=v_model, width=12, state='readonly',
        values=['tiny','base','small','medium','large','large-v3','turbo']
    ).grid(row=2, column=1, sticky='w', padx=4)

    tk.Label(f, text="Language", anchor='w', width=20).grid(row=3, column=0, sticky='w', padx=6, pady=3)
    ttk.Combobox(f, textvariable=v_lang, width=8, state='readonly',
        values=['fr','en','es','de','it','pt','pl','tr','ru','nl','cs','ar','zh','hu','ko','ja','hi']
    ).grid(row=3, column=1, sticky='w', padx=4)

    try:
        import torch as _torch_txt
        _txt_dev = "cuda" if _torch_txt.cuda.is_available() else "cpu"
    except Exception:
        _txt_dev = "cpu"
    v_device = tk.StringVar(value=_txt_dev)
    tk.Label(f, text="Device", anchor='w', width=20).grid(row=4, column=0, sticky='w', padx=6, pady=3)
    ttk.Combobox(f, textvariable=v_device, width=8, state='readonly',
        values=['cpu','cuda']).grid(row=4, column=1, sticky='w', padx=4)

    tk.Label(f, text="Min pause (s)", anchor='w', width=20).grid(row=5, column=0, sticky='w', padx=6, pady=3)
    tk.Entry(f, textvariable=v_pause, width=8).grid(row=5, column=1, sticky='w', padx=4)

    tk.Checkbutton(f, text="Pitch annotation [p:±N]", variable=v_pitch).grid(
        row=6, column=1, sticky='w', padx=4, pady=3)

    console = add_console(f, 7)

    def lancer(btn, stop_btn=None):
        if not v_input.get() or not v_output.get():
            log(console, "[ERR] Source and output required."); return
        ext = os.path.splitext(v_input.get())[1].lower()
        video_exts = {'.mp4','.mkv','.avi','.mov','.flv','.webm','.wmv','.m4v','.ts','.mpg'}
        if ext in video_exts:
            cmd = [sys.executable, os.path.join(SCRIPTS_DIR, 'video2txt.py'),
                   v_input.get(), v_output.get(),
                   '--model', v_model.get(), '--lang', v_lang.get(),
                   '--pause', v_pause.get(), '--device', v_device.get()]
            if v_pitch.get(): cmd.append('--pitch')
        else:
            cmd = [sys.executable, os.path.join(SCRIPTS_DIR, 'transcribeSong2txt_with_pause.py'),
                   v_input.get(), v_output.get(),
                   v_model.get(), v_pause.get(), v_lang.get(), '--device', v_device.get()]
            if v_pitch.get(): cmd.append('--pitch')
        run_cmd(cmd, console, btn, stop_btn)

    make_btn(f, ">  Transcribe", lancer, 6)


# ── Tab: Voice Separation ───────────────────────────────────────────────────

def tab_extract(nb):
    f = ttk.Frame(nb)
    nb.add(f, text="[Vox] Voice sep.")
    f.grid_columnconfigure(1, weight=1)

    v_input      = tk.StringVar()
    v_output     = tk.StringVar()
    v_keep       = tk.StringVar(value='female')
    v_silence    = tk.StringVar(value='auto')
    v_thr        = tk.StringVar(value='165')
    v_deverb     = tk.StringVar(value='none')
    v_debug      = tk.BooleanVar(value=False)
    v_split      = tk.BooleanVar(value=False)
    v_ovrange    = tk.StringVar(value='200')
    v_minsilence = tk.StringVar(value='0.30')
    v_mindur     = tk.StringVar(value='0.2')
    v_remove_music = tk.BooleanVar(value=False)
    v_demucs_model = tk.StringVar(value='htdemucs_ft')
    v_mp3_bitrate  = tk.StringVar(value='192')
    v_mp3_mode     = tk.StringVar(value='cbr')
    v_method       = tk.StringVar(value='f0')

    # row 0 : source
    add_row(f, "Audio/Video source", v_input, 0,
            [("Audio/Video","*.wav *.mp3 *.flac *.ogg *.mp4 *.mkv *.avi *.mov *.webm *.m4a"),("All","*.*")],
            initialdir=DIR_VOICES)
    # row 1 : output
    add_row(f, "Output (wav/mp3)", v_output, 1,
            [("WAV","*.wav"),("MP3","*.mp3"),("FLAC","*.flac"),("OGG","*.ogg"),("All","*.*")],
            save=True, initialdir=DIR_OUTPUT)

    # row 2 : Keep + Split
    tk.Label(f, text="Keep", anchor='w', width=20).grid(row=2, column=0, sticky='w', padx=6, pady=3)
    frm_keep = tk.Frame(f)
    frm_keep.grid(row=2, column=1, sticky='w', padx=4)
    ttk.Combobox(frm_keep, textvariable=v_keep, width=14, state='readonly',
        values=['female','male','overlap','all','female,male','vocals only']).pack(side='left')
    tk.Checkbutton(frm_keep, text="Split F+M  (genere _female + _male)",
                   variable=v_split).pack(side='left', padx=10)

    # row 3 : Silence
    tk.Label(f, text="Silence (s/auto/0)", anchor='w', width=20).grid(row=3, column=0, sticky='w', padx=6, pady=3)
    tk.Entry(f, textvariable=v_silence, width=8).grid(row=3, column=1, sticky='w', padx=4)

    # row 4 : F0 threshold
    tk.Label(f, text="F0 threshold (Hz)", anchor='w', width=20).grid(row=4, column=0, sticky='w', padx=6, pady=3)
    frm_thr = tk.Frame(f)
    frm_thr.grid(row=4, column=1, sticky='w', padx=4)
    tk.Entry(frm_thr, textvariable=v_thr, width=8).pack(side='left')
    tk.Label(frm_thr, text="  (femme >= seuil, homme < seuil)", fg='grey').pack(side='left')

    # row 5 : Overlap range
    tk.Label(f, text="Overlap range (Hz)", anchor='w', width=20).grid(row=5, column=0, sticky='w', padx=6, pady=3)
    frm_ov = tk.Frame(f)
    frm_ov.grid(row=5, column=1, sticky='w', padx=4)
    tk.Entry(frm_ov, textvariable=v_ovrange, width=8).pack(side='left')
    tk.Label(frm_ov, text="  (augmenter si voix H classees overlap)", fg='grey').pack(side='left')

    # row 6 : Min silence
    tk.Label(f, text="Min silence (s)", anchor='w', width=20).grid(row=6, column=0, sticky='w', padx=6, pady=3)
    tk.Entry(f, textvariable=v_minsilence, width=8).grid(row=6, column=1, sticky='w', padx=4)

    # row 7 : Min dur segment
    tk.Label(f, text="Min dur segment (s)", anchor='w', width=20).grid(row=7, column=0, sticky='w', padx=6, pady=3)
    frm_mindur = tk.Frame(f)
    frm_mindur.grid(row=7, column=1, sticky='w', padx=4)
    tk.Entry(frm_mindur, textvariable=v_mindur, width=8).pack(side='left')
    tk.Label(frm_mindur, text="  (0.5-1.0 pour ignorer les bribes)", fg='grey').pack(side='left')

    # row 8 : Dereverberation
    tk.Label(f, text="Dereverberation", anchor='w', width=20).grid(row=8, column=0, sticky='w', padx=6, pady=3)
    ttk.Combobox(f, textvariable=v_deverb, width=14, state='readonly',
        values=['none','noisereduce','wpe','deepfilter']).grid(row=8, column=1, sticky='w', padx=4)

    # row 9 : Debug
    frm_checks = tk.Frame(f)
    frm_checks.grid(row=9, column=0, columnspan=3, sticky='w', padx=6, pady=2)
    tk.Checkbutton(frm_checks, text="Debug mode", variable=v_debug).pack(side='left', padx=6)

    # row 10 : Device
    try:
        import torch as _torch_vox
        _vox_default = "cuda" if _torch_vox.cuda.is_available() else "cpu"
    except Exception:
        _vox_default = "cpu"
    v_vox_device = tk.StringVar(value=_vox_default)
    tk.Label(f, text="Device", anchor='w', width=20).grid(row=10, column=0, sticky='w', padx=6, pady=3)
    ttk.Combobox(f, textvariable=v_vox_device, width=8, state='readonly',
        values=['cpu', 'cuda']).grid(row=10, column=1, sticky='w', padx=4)

    v_vox_tempo = tk.StringVar(value='1.0')
    tk.Label(f, text="Tempo × (pitch kept)", anchor='w', width=20).grid(row=15, column=0, sticky='w', padx=6, pady=3)
    ttk.Combobox(f, textvariable=v_vox_tempo, width=8,
        values=['0.70','0.80','0.85','0.90','1.0','1.10','1.25','1.5']).grid(row=15, column=1, sticky='w', padx=4)

    # row 11 : Remove music
    def on_remove_music_toggle(*args):
        if v_remove_music.get():
            v_keep.set('vocals only')
        else:
            if v_keep.get() == 'vocals only':
                v_keep.set('female')
    v_remove_music.trace_add('write', on_remove_music_toggle)

    tk.Checkbutton(f, text="Remove background music (demucs)",
                   variable=v_remove_music).grid(row=11, column=0, columnspan=2, sticky='w', padx=6, pady=2)

    # row 12 : Demucs model + shifts (quality passes)
    tk.Label(f, text="Demucs model", anchor='w', width=20).grid(row=12, column=0, sticky='w', padx=6, pady=3)
    frm_dm = tk.Frame(f)
    frm_dm.grid(row=12, column=1, sticky='w', padx=4)
    ttk.Combobox(frm_dm, textvariable=v_demucs_model, width=16, state='readonly',
        values=['htdemucs', 'htdemucs_ft', 'mdx_extra']).pack(side='left')
    v_demucs_shifts = tk.StringVar(value='2')
    tk.Label(frm_dm, text="Shifts").pack(side='left', padx=(10, 2))
    ttk.Combobox(frm_dm, textvariable=v_demucs_shifts, width=4, state='readonly',
        values=['1', '2', '5']).pack(side='left')
    tk.Label(frm_dm, text="(1=fast, 5=best)", fg='grey').pack(side='left', padx=(4, 0))

    # row 13 : MP3 bitrate
    tk.Label(f, text="MP3 bitrate (kbps)", anchor='w', width=20).grid(row=13, column=0, sticky='w', padx=6, pady=3)
    frm_mp3 = tk.Frame(f)
    frm_mp3.grid(row=13, column=1, sticky='w', padx=4)
    ttk.Combobox(frm_mp3, textvariable=v_mp3_bitrate, width=6, state='readonly',
        values=['128','160','192','256','320']).pack(side='left')
    ttk.Combobox(frm_mp3, textvariable=v_mp3_mode, width=5, state='readonly',
        values=['cbr','vbr']).pack(side='left', padx=6)
    tk.Label(frm_mp3, text="(only used if output is .mp3)", fg='grey').pack(side='left')

    # row 14 : Methode
    tk.Label(f, text="Methode", anchor='w', width=20).grid(row=14, column=0, sticky='w', padx=6, pady=3)
    frm_method = tk.Frame(f)
    frm_method.grid(row=14, column=1, sticky='w', padx=4)
    ttk.Combobox(frm_method, textvariable=v_method, width=12, state='readonly',
        values=['f0', 'ecapa', 'sepformer', 'pyannote']).pack(side='left')
    tk.Label(frm_method, text="(ecapa = by speaker, not pitch)", fg='grey').pack(side='left', padx=(6, 0))
    tk.Label(frm_method,
             text="  f0=rapide  |  sepformer=separation reelle  |  pyannote=diarisation",
             fg='grey').pack(side='left')

    # row 15 : console
    console = add_console(f, 16)

    def _build_cmd(with_output=True):
        cmd = [sys.executable, os.path.join(SCRIPTS_DIR, 'extract_voices.py'),
               v_input.get()]
        if with_output:
            cmd.append(v_output.get())
        cmd += ['--keep',          v_keep.get(),
                '--silence',       v_silence.get(),
                '--threshold',     v_thr.get(),
                '--overlap-range', v_ovrange.get(),
                '--min-silence',   v_minsilence.get(),
                '--min-dur',       v_mindur.get(),
                '--dereverberate', v_deverb.get(),
                '--method',        v_method.get()]
        if v_debug.get():
            cmd.append('--debug')
        cmd += ['--device', v_vox_device.get()]
        if v_vox_tempo.get().strip() not in ('', '1.0', '1'):
            cmd += ['--tempo', v_vox_tempo.get().strip()]
        cmd += ['--mp3-bitrate', v_mp3_bitrate.get(), '--mp3-mode', v_mp3_mode.get()]
        if v_remove_music.get():
            cmd += ['--remove-music', '--demucs-model', v_demucs_model.get(),
                    '--demucs-shifts', v_demucs_shifts.get()]
        return cmd

    def lancer(btn, stop_btn=None):
        if not v_input.get() or not v_output.get():
            log(console, "[ERR] Source and output required."); return
        cmd = _build_cmd(with_output=True)
        if v_split.get():
            cmd.append('--split-output')
        run_cmd(cmd, console, btn, stop_btn)

    def analyser(btn, stop_btn=None):
        if not v_input.get():
            log(console, "[ERR] Source required."); return
        cmd = _build_cmd(with_output=False) + ['--analyze']
        run_cmd(cmd, console, btn, stop_btn)

    btn_sep, stop_sep = make_btn(f, ">  Separate", lancer,   17)
    btn_ana, stop_ana = make_btn(f, "[?] Analyze", analyser, 18)
    btn_ana.config(bg='#7d5a2d')


# ── Tab: Pitch ──────────────────────────────────────────────────────────────

def tab_pitch(nb):
    f = ttk.Frame(nb)
    nb.add(f, text="[Pit] Pitch")
    f.grid_columnconfigure(1, weight=1)

    v_clone  = tk.StringVar()
    v_txt    = tk.StringVar()
    v_output = tk.StringVar()
    v_shift  = tk.StringVar(value='0')
    v_lang   = tk.StringVar(value='fr')
    v_model  = tk.StringVar(value='small')

    add_row(f, "Clone (.wav)",        v_clone,  0, [("WAV","*.wav")], initialdir=DIR_VOICES)
    add_row(f, "Script (.txt)",       v_txt,    1, [("Text","*.txt")])
    add_row(f, "Output (.wav)",       v_output, 2, [("WAV","*.wav")], save=True, initialdir=DIR_OUTPUT)

    tk.Label(f, text="Global shift (st)", anchor='w', width=20).grid(row=3, column=0, sticky='w', padx=6, pady=3)
    tk.Entry(f, textvariable=v_shift, width=8).grid(row=3, column=1, sticky='w', padx=4)

    tk.Label(f, text="Language", anchor='w', width=20).grid(row=4, column=0, sticky='w', padx=6, pady=3)
    ttk.Combobox(f, textvariable=v_lang, width=8, state='readonly',
        values=['fr','en','es','de','it']
    ).grid(row=4, column=1, sticky='w', padx=4)

    tk.Label(f, text="Whisper model", anchor='w', width=20).grid(row=5, column=0, sticky='w', padx=6, pady=3)
    ttk.Combobox(f, textvariable=v_model, width=10, state='readonly',
        values=['tiny','base','small','medium']
    ).grid(row=5, column=1, sticky='w', padx=4)

    try:
        import torch as _torch_pit
        _pit_dev = "cuda" if _torch_pit.cuda.is_available() else "cpu"
    except Exception:
        _pit_dev = "cpu"
    v_device = tk.StringVar(value=_pit_dev)
    tk.Label(f, text="Device", anchor='w', width=20).grid(row=6, column=0, sticky='w', padx=6, pady=3)
    ttk.Combobox(f, textvariable=v_device, width=8, state='readonly',
        values=['cpu','cuda']).grid(row=6, column=1, sticky='w', padx=4)

    console = add_console(f, 8)

    def lancer(btn, stop_btn=None):
        if not v_clone.get() or not v_output.get():
            log(console, "[ERR] Clone and output required."); return
        cmd = [sys.executable, os.path.join(SCRIPTS_DIR, 'apply_pitch_to_clone.py'),
               v_clone.get(), v_txt.get(), v_output.get(),
               '--global-shift', v_shift.get(),
               '--lang', v_lang.get(),
               '--model', v_model.get(),
               '--device', v_device.get()]
        run_cmd(cmd, console, btn, stop_btn)

    make_btn(f, ">  Apply pitch", lancer, 6)


# ── Tab: Video to MP3 ───────────────────────────────────────────────────────

def tab_convert(nb):
    f = ttk.Frame(nb)
    nb.add(f, text="[Vid] Video->MP3")
    f.grid_columnconfigure(1, weight=1)

    v_input  = tk.StringVar()
    v_output = tk.StringVar()

    add_row(f, "Video source",  v_input,  0,
            [("Video","*.mp4 *.mkv *.avi *.mov *.flv *.webm *.wmv"),("All","*.*")])
    add_row(f, "Output (mp3/wav/flac/ogg)", v_output, 1,
            [("MP3","*.mp3"),("WAV","*.wav"),("FLAC","*.flac"),("OGG","*.ogg"),("All","*.*")],
            save=True, initialdir=DIR_OUTPUT)

    # ── Audio options ─────────────────────────────────────────────────────
    v_vid_mp3_bitrate = tk.StringVar(value='192')
    v_vid_mp3_mode    = tk.StringVar(value='cbr')
    v_vid_channels    = tk.StringVar(value='stereo')
    v_vid_samplerate  = tk.StringVar(value='44100')
    v_vid_tempo       = tk.StringVar(value='1.0')

    # Row 2: MP3 bitrate
    tk.Label(f, text="MP3 bitrate (kbps)", anchor='w', width=20).grid(row=2, column=0, sticky='w', padx=6, pady=3)
    frm_vid_mp3 = tk.Frame(f)
    frm_vid_mp3.grid(row=2, column=1, sticky='w', padx=4)
    ttk.Combobox(frm_vid_mp3, textvariable=v_vid_mp3_bitrate, width=6, state='readonly',
        values=['128','160','192','256','320']).pack(side='left')
    ttk.Combobox(frm_vid_mp3, textvariable=v_vid_mp3_mode, width=5, state='readonly',
        values=['cbr','vbr']).pack(side='left', padx=6)
    tk.Label(frm_vid_mp3, text="(only used if output is .mp3)", fg='grey').pack(side='left')

    # Row 3: Channels + Sample rate + XTTS preset
    tk.Label(f, text="Channels", anchor='w', width=20).grid(row=3, column=0, sticky='w', padx=6, pady=3)
    frm_vid_audio = tk.Frame(f)
    frm_vid_audio.grid(row=3, column=1, sticky='w', padx=4)
    ttk.Combobox(frm_vid_audio, textvariable=v_vid_channels, width=8, state='readonly',
        values=['stereo','mono']).pack(side='left')
    tk.Label(frm_vid_audio, text="Sample rate (Hz)").pack(side='left', padx=(12,4))
    ttk.Combobox(frm_vid_audio, textvariable=v_vid_samplerate, width=8, state='readonly',
        values=['16000','22050','44100','48000']).pack(side='left')

    def _xtts_preset():
        # Force WAV output path extension
        p = v_output.get()
        if p:
            base = os.path.splitext(p)[0]
            v_output.set(base + '.wav')
        v_vid_channels.set('mono')
        v_vid_samplerate.set('22050')

    tk.Button(frm_vid_audio, text="XTTS preset", bg='#1a6b9e', fg='white',
              command=_xtts_preset).pack(side='left', padx=10)

    # Row 4: Tempo (pitch preserved) on its own line
    tk.Label(f, text="Tempo ×", anchor='w', width=20).grid(row=4, column=0, sticky='w', padx=6, pady=3)
    frm_vid_tempo = tk.Frame(f)
    frm_vid_tempo.grid(row=4, column=1, sticky='w', padx=4)
    ttk.Combobox(frm_vid_tempo, textvariable=v_vid_tempo, width=6,
        values=['0.70','0.80','0.85','0.90','1.0','1.10','1.25','1.5']).pack(side='left')
    tk.Label(frm_vid_tempo, text="(pitch preserved — slower < 1.0 < faster)", fg='grey').pack(side='left', padx=(6,0))

    console = add_console(f, 6)

    def lancer(btn, stop_btn=None):
        if not v_input.get() or not v_output.get():
            log(console, "[ERR] Source and output required."); return
        ext = os.path.splitext(v_output.get())[1].lower()
        ch  = '1' if v_vid_channels.get() == 'mono' else '2'
        sr  = v_vid_samplerate.get()

        base_args = ['ffmpeg', '-y', '-i', v_input.get(), '-vn',
                     '-ac', ch, '-ar', sr]

        # Tempo (time-stretch) preserving pitch/timbre, via ffmpeg atempo.
        # atempo is valid for 0.5–2.0, so chain factors for anything outside.
        try:
            tempo = float(v_vid_tempo.get())
        except ValueError:
            tempo = 1.0
        if abs(tempo - 1.0) > 1e-3 and tempo > 0:
            t, parts = tempo, []
            while t < 0.5:
                parts.append(0.5); t /= 0.5
            while t > 2.0:
                parts.append(2.0); t /= 2.0
            parts.append(t)
            chain = ','.join(f'atempo={p:.4f}' for p in parts)
            base_args += ['-filter:a', chain]
            log(console, f"[*] Tempo ×{tempo} (pitch preserved): {chain}")

        if ext == '.mp3':
            br = v_vid_mp3_bitrate.get()
            if v_vid_mp3_mode.get() == 'vbr':
                vbr_map = {'128':'6','160':'5','192':'4','256':'2','320':'0'}
                cmd = base_args + ['-codec:a', 'libmp3lame', '-q:a',
                                   vbr_map.get(br,'4'), v_output.get()]
            else:
                cmd = base_args + ['-codec:a', 'libmp3lame', '-b:a',
                                   f'{br}k', v_output.get()]
        elif ext == '.flac':
            cmd = base_args + ['-codec:a', 'flac', v_output.get()]
        elif ext == '.ogg':
            cmd = base_args + ['-codec:a', 'libvorbis', v_output.get()]
        else:  # .wav
            cmd = base_args + ['-sample_fmt', 's16', v_output.get()]
        run_cmd(cmd, console, btn, stop_btn)

    make_btn(f, ">  Convert", lancer, 5)



# ── Tab: Validator ──────────────────────────────────────────────────────────

def tab_validator(nb):
    f = tk.Frame(nb)
    nb.add(f, text="[Val] Validator")

    v_val_voices  = tk.StringVar()
    v_val_lang    = tk.StringVar(value='FR')
    v_val_param   = tk.StringVar(value='seed')
    v_val_values  = tk.StringVar(value='0 7 13 42 100 200')
    v_val_text    = tk.StringVar(value="Bonjour, ceci est un test de validation de la voix.")
    v_val_output  = tk.StringVar()
    v_val_xtts    = tk.StringVar()   # {} block from voice_analyser
    v_val_audio   = tk.StringVar()   # [] block from voice_analyser

    row = 0
    # Voice refs
    tk.Label(f, text="Voice ref(s)", anchor='w', width=18).grid(row=row, column=0, sticky='w', padx=6, pady=3)
    frm_v = tk.Frame(f); frm_v.grid(row=row, column=1, sticky='ew', padx=4)
    f.grid_columnconfigure(1, weight=1)
    frm_v.grid_columnconfigure(0, weight=1)
    tk.Entry(frm_v, textvariable=v_val_voices).grid(row=0, column=0, sticky='ew')
    def browse_val_voices():
        files = filedialog.askopenfilenames(
            filetypes=[("Audio","*.wav *.mp3 *.flac *.ogg"),("All","*.*")],
            initialdir=DIR_VOICES)
        if files:
            existing = v_val_voices.get().strip()
            new_files = " ".join(files)
            v_val_voices.set((existing + " " + new_files).strip() if existing else new_files)
    tk.Button(frm_v, text="Browse", width=8, command=browse_val_voices).grid(row=0, column=1, padx=2)

    row += 1
    tk.Label(f, text="Language", anchor='w', width=18).grid(row=row, column=0, sticky='w', padx=6, pady=3)
    frm_lang = tk.Frame(f)
    frm_lang.grid(row=row, column=1, sticky='w', padx=4)
    ttk.Combobox(frm_lang, textvariable=v_val_lang, values=LANGS, width=8, state='readonly').pack(side='left')

    v_fill_mode = tk.StringVar(value='default')
    tk.Radiobutton(frm_lang, text="Default", variable=v_fill_mode, value='default').pack(side='left', padx=(10,2))
    tk.Radiobutton(frm_lang, text="Raw",     variable=v_fill_mode, value='raw').pack(side='left', padx=2)

    def _fill_blocks():
        lang = v_val_lang.get()
        mode = v_fill_mode.get()
        if mode == 'default':
            # Generator defaults
            xtts_str  = "{1, 0, 0, 0, 100, 250, 0.65, 50, 0.85, 5.0, 1.0, 30, 4, 0}"
            audio_str = f"[1, {lang}, 0.9, 3, -2, 3, -4, 90, 8000, 0.5, 0.4, 0.3, 0, 0, 0, 0]"
        else:  # raw — native XTTS defaults, no audio processing
            xtts_str  = "{1, 0, 0, 0, 0, 0, 0.75, 50, 0.85, 10.0, 1.0, 30, 4, 0}"
            audio_str = f"[1, {lang}, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]"
        v_val_xtts.set(xtts_str)
        v_val_audio.set(audio_str)

    tk.Button(frm_lang, text="Fill", width=5, command=_fill_blocks).pack(side='left', padx=6)

    row += 1
    tk.Label(f, text="XTTS params {}", anchor='w', width=18).grid(row=row, column=0, sticky='w', padx=6, pady=3)
    tk.Label(f, text="{N, seed, trim_start, trim_end, fade_in, fade_out, temp, top_k, top_p, rep_pen, len_pen, gpt_cond_len, gpt_cond_chunk_len, sound_norm_refs}",
             fg='gray', font=('Arial',8)).grid(row=row, column=1, sticky='w', padx=4)
    row += 1
    tk.Entry(f, textvariable=v_val_xtts).grid(row=row, column=1, sticky='ew', padx=4)

    row += 1
    tk.Label(f, text="Audio params []", anchor='w', width=18).grid(row=row, column=0, sticky='w', padx=6, pady=3)
    tk.Label(f, text="[N, LANG, speed, vol, eq_low, eq_mid, eq_high, hp, lp, NR, comp, de-ess, reverb, noise_gate, pan, limiter]",
             fg='gray', font=('Arial',8)).grid(row=row, column=1, sticky='w', padx=4)
    row += 1
    tk.Entry(f, textvariable=v_val_audio).grid(row=row, column=1, sticky='ew', padx=4)

    ALL_PARAMS = ['seed','temp','top_k','top_p','rep_pen','len_pen','gpt_cond_len',
                  'speed','vol','eq_low','eq_mid','eq_high','hp','lp','NR','comp','de-ess','reverb','noise_gate','pan']

    row += 1
    tk.Label(f, text="Parameters", anchor='w', width=18).grid(row=row, column=0, sticky='nw', padx=6, pady=3)
    tk.Label(f, text="XTTS: seed temp top_k top_p rep_pen len_pen  |  Audio: speed vol eq_low eq_mid eq_high hp lp NR comp de-ess reverb noise_gate pan",
             fg='gray', font=('Arial',8)).grid(row=row, column=1, sticky='w', padx=4)

    # Dynamic param rows frame
    params_frame = tk.Frame(f)
    params_frame.grid(row=row, column=1, sticky='ew', padx=4, pady=(18,0))
    params_frame.grid_columnconfigure(1, weight=1)
    param_rows = []  # list of (StringVar_param, StringVar_values, Frame)

    lbl_total = tk.Label(f, text="Total: 1 combination", fg='gray', font=('Arial',8))

    VAL_DEFAULTS = {
        'seed': 42, 'temp': 0.72, 'top_k': 50, 'top_p': 0.85,
        'rep_pen': 5.0, 'len_pen': 1.0, 'gpt_cond_len': 30,
        'speed': 1.0, 'vol': 0, 'eq_low': 0, 'eq_mid': 0, 'eq_high': 0,
        'hp': 0, 'lp': 0, 'NR': 0, 'comp': 0, 'de-ess': 0,
        'reverb': 0, 'noise_gate': 0, 'pan': 0,
    }

    def _parse_blocks():
        import re as _re
        params = VAL_DEFAULTS.copy()
        xtts = v_val_xtts.get().strip()
        audio = v_val_audio.get().strip()
        if xtts:
            try:
                nums = [float(x) for x in _re.findall(r'[-+]?\d+(?:\.\d+)?', xtts)]
                keys = ['seed','trim_start','trim_end','fade_in','fade_out',
                        'temp','top_k','top_p','rep_pen','len_pen',
                        'gpt_cond_len','gpt_cond_chunk_len','sound_norm_refs']
                for i, k in enumerate(keys):
                    if i+1 < len(nums): params[k] = nums[i+1]
            except Exception: pass
        if audio:
            try:
                all_nums = [float(x) for x in _re.findall(r'[-+]?\d+(?:\.\d+)?', audio)]
                numeric_vals = all_nums[1:]  # skip N
                keys = ['speed','vol','eq_low','eq_mid','eq_high',
                        'hp','lp','NR','comp','de-ess','reverb','noise_gate','pan']
                for i, k in enumerate(keys):
                    if i < len(numeric_vals): params[k] = numeric_vals[i]
            except Exception: pass
        return params

    def _update_total():
        total = 1
        for vp, vv, _ in param_rows:
            vals = [x for x in vv.get().split() if x.strip()]
            if vals:
                total *= len(vals)
        lbl_total.config(text=f"Total: {total} combination{'s' if total>1 else ''}")

    def _refresh_comboboxes():
        """Update each combobox to exclude params used in other rows."""
        for entry in param_rows:
            vp, vv, row_f = entry
            current = vp.get()
            used_elsewhere = [r[0].get() for r in param_rows if r is not entry]
            available = [p for p in ALL_PARAMS if p not in used_elsewhere]
            # Find the combobox widget in this row
            for child in row_f.winfo_children():
                if isinstance(child, ttk.Combobox):
                    child['values'] = available
                    break

    def _add_param_row(param=None, values_str=None):
        # Pick first unused param if none specified
        if param is None:
            used = [r[0].get() for r in param_rows]
            available = [p for p in ALL_PARAMS if p not in used]
            param = available[0] if available else ALL_PARAMS[0]

        vp = tk.StringVar(value=param)
        vv = tk.StringVar(value=values_str or '')

        row_f = tk.Frame(params_frame)
        row_f.pack(fill='x', pady=1, before=btn_add_p)
        row_f.grid_columnconfigure(1, weight=1)

        cb = ttk.Combobox(row_f, textvariable=vp, values=ALL_PARAMS, width=12, state='readonly')
        cb.pack(side='left', padx=2)

        entry = tk.Entry(row_f, textvariable=vv, width=30)
        entry.pack(side='left', fill='x', expand=True, padx=4)
        vv.trace_add('write', lambda *a: _update_total())

        entry_tuple = (vp, vv, row_f)

        def _on_select(event=None, _vp=vp, _vv=vv):
            p = _vp.get()
            params = _parse_blocks()
            val = params.get(p, VAL_DEFAULTS.get(p, 0))
            val_str = str(int(val)) if isinstance(val, float) and val == int(val) else str(val)
            _vv.set(val_str)
            _update_total()
            _refresh_comboboxes()

        cb.bind('<<ComboboxSelected>>', _on_select)

        def remove(e=entry_tuple):
            e[2].destroy()
            param_rows.remove(e)
            _update_total()
            _refresh_comboboxes()

        tk.Button(row_f, text="X", width=2, fg='red', command=remove).pack(side='left', padx=1)
        param_rows.append(entry_tuple)
        _update_total()
        _refresh_comboboxes()

    btn_add_p = tk.Button(params_frame, text="+ Add parameter", command=lambda: _add_param_row(),
                          bg='#2d6a2d', fg='white')
    btn_add_p.pack(anchor='w', pady=2)

    _add_param_row(param='seed')

    lbl_total.grid(row=row, column=2, sticky='w', padx=4)

    row += 1
    tk.Label(f, text="Text", anchor='w', width=18).grid(row=row, column=0, sticky='w', padx=6, pady=3)
    tk.Entry(f, textvariable=v_val_text).grid(row=row, column=1, sticky='ew', padx=4)

    row += 1
    tk.Label(f, text="Output (wav)", anchor='w', width=18).grid(row=row, column=0, sticky='w', padx=6, pady=3)
    frm_o = tk.Frame(f); frm_o.grid(row=row, column=1, sticky='ew', padx=4)
    frm_o.grid_columnconfigure(0, weight=1)
    tk.Entry(frm_o, textvariable=v_val_output).grid(row=0, column=0, sticky='ew')
    tk.Button(frm_o, text="Browse", width=8,
        command=lambda: browse_file(v_val_output, [("WAV","*.wav")], save=True, initialdir=DIR_OUTPUT)
    ).grid(row=0, column=1, padx=2)

    row += 1
    console = add_console(f, row)

    def lancer(btn, stop_btn=None):
        refs = [r for r in v_val_voices.get().split() if r.strip()]
        if not refs:
            log(console, "[ERR] At least one voice reference required."); return
        output = v_val_output.get().strip()

        # Collect param rows — rows with empty values generate a single base variation
        valid_rows = [(vp.get(), vv.get().strip()) for vp, vv, _ in param_rows if vv.get().strip()]
        if not valid_rows:
            # No values specified — generate a single variation with base params
            valid_rows = [(param_rows[0][0].get(), '_base')] if param_rows else []
            if not valid_rows:
                log(console, "[ERR] At least one parameter row required."); return

        if not output:
            params_str = '_'.join(p for p, v in valid_rows)
            output = os.path.join(DIR_OUTPUT, f"validation_{params_str}.wav")
            v_val_output.set(output)

        cmd = [sys.executable, os.path.join(SCRIPTS_DIR, 'voice_validator.py')]
        cmd += refs
        cmd += [v_val_lang.get()]
        # Pass each param row as --param name --values v1 v2 v3
        for param_name, values_str in valid_rows:
            cmd += ['--param', param_name, '--values'] + values_str.split()
        cmd += ['--text', v_val_text.get()]
        cmd += ['--output', output]
        if v_val_xtts.get().strip():
            cmd += ['--xtts-block', v_val_xtts.get().strip()]
        if v_val_audio.get().strip():
            cmd += ['--audio-block', v_val_audio.get().strip()]

        def _val_on_line(txt):
            # Capture the winning {} block printed after "Paste into the comparator"
            s = txt.strip()
            if s.startswith('{1,') or s.startswith('{1 '):
                HANDOFF['win_xtts'] = s
        run_cmd(cmd, console, btn, stop_btn, line_callback=_val_on_line)

    make_btn(f, ">  Generate", lancer, row + 1)

    # Register as a target (Analyser -> Validator) and add hand-off to Comparator
    TARGETS['val_xtts']  = v_val_xtts
    TARGETS['val_audio'] = v_val_audio
    frm_vho = tk.Frame(f)
    frm_vho.grid(row=row + 2, column=0, columnspan=3, sticky='w', padx=6, pady=(0, 4))
    tk.Label(frm_vho, text="Send winning {} →", fg="gray", font=("Arial", 8)).pack(side='left', padx=(0, 4))
    def _val_to_cmp():
        block = HANDOFF.get('win_xtts') or v_val_xtts.get().strip()
        ok = _handoff_set('cmp_xtts', block)
        log(console, "[→] Sent winning {} to Comparator" if ok else "[!] Run a sweep first.")
    tk.Button(frm_vho, text="→ Comparator", command=_val_to_cmp).pack(side='left', padx=2)


# ── Tab: Optimiser ────────────────────────────────────────────────────────────

def tab_optimize(nb):
    f = ttk.Frame(nb)
    nb.add(f, text="[Opt] Optimiser")
    f.grid_columnconfigure(1, weight=1)

    v_opt_voices = tk.StringVar()
    v_opt_lang   = tk.StringVar(value='FR')
    v_opt_xtts   = tk.StringVar()
    v_opt_text   = tk.StringVar(value="Bonjour, ceci est une phrase de test pour régler la voix avec soin.")
    v_opt_seeds  = tk.StringVar(value='0 42 100 200')
    v_opt_budget = tk.StringVar(value='60')
    v_opt_wacc   = tk.StringVar(value='0.6')
    v_opt_wid    = tk.StringVar(value='0.4')
    v_opt_maxref = tk.StringVar(value='30')
    v_opt_model  = tk.StringVar(value='small')

    TARGETS['opt_xtts'] = v_opt_xtts   # Analyser -> Optimiser hand-off
    TARGETS['opt_voices'] = v_opt_voices   # Curation -> Optimiser hand-off

    add_row(f, "Voice ref(s)", v_opt_voices, 0,
            [("Audio", "*.wav *.mp3 *.flac *.ogg"), ("All", "*.*")],
            multi=True, initialdir=DIR_VOICES)

    tk.Label(f, text="Language", anchor='w', width=20).grid(row=1, column=0, sticky='w', padx=6, pady=3)
    ttk.Combobox(f, textvariable=v_opt_lang, values=LANGS, width=8, state='readonly').grid(row=1, column=1, sticky='w', padx=4)

    tk.Label(f, text="XTTS params {}", anchor='w', width=20).grid(row=2, column=0, sticky='w', padx=6, pady=3)
    tk.Entry(f, textvariable=v_opt_xtts).grid(row=2, column=1, sticky='ew', padx=4)

    tk.Label(f, text="Test text", anchor='w', width=20).grid(row=3, column=0, sticky='w', padx=6, pady=3)
    tk.Entry(f, textvariable=v_opt_text).grid(row=3, column=1, sticky='ew', padx=4)

    tk.Label(f, text="Seeds to screen", anchor='w', width=20).grid(row=4, column=0, sticky='w', padx=6, pady=3)
    tk.Entry(f, textvariable=v_opt_seeds).grid(row=4, column=1, sticky='ew', padx=4)

    frm_n = tk.Frame(f); frm_n.grid(row=5, column=0, columnspan=3, sticky='w', padx=6, pady=3)
    for lbl, var, w in [("Budget", v_opt_budget, 5), ("w_accent", v_opt_wacc, 5),
                        ("w_identity", v_opt_wid, 5), ("max_ref_len", v_opt_maxref, 5)]:
        tk.Label(frm_n, text=lbl).pack(side='left', padx=(6, 2))
        tk.Entry(frm_n, textvariable=var, width=w).pack(side='left')
    tk.Label(frm_n, text="whisper").pack(side='left', padx=(10, 2))
    ttk.Combobox(frm_n, textvariable=v_opt_model, values=['tiny', 'base', 'small', 'medium'],
                 width=8, state='readonly').pack(side='left')

    try:
        import torch as _t_opt
        _opt_dev = "cuda" if _t_opt.cuda.is_available() else "cpu"
    except Exception:
        _opt_dev = "cpu"
    v_opt_device = tk.StringVar(value=_opt_dev)
    tk.Label(f, text="Device", anchor='w', width=20).grid(row=6, column=0, sticky='w', padx=6, pady=3)
    ttk.Combobox(f, textvariable=v_opt_device, values=['cpu', 'cuda'], width=8, state='readonly').grid(row=6, column=1, sticky='w', padx=4)

    console = add_console(f, 8)

    def lancer(btn, stop_btn=None):
        refs = [r for r in v_opt_voices.get().split() if r.strip()]
        if not refs:
            log(console, "[ERR] At least one voice reference required."); return
        if not v_opt_xtts.get().strip():
            log(console, "[ERR] XTTS {} block required (send it from the Analyser)."); return
        cmd = [sys.executable, os.path.join(SCRIPTS_DIR, 'xtts_optimize.py')]
        cmd += refs
        cmd += [v_opt_lang.get()]
        cmd += ['--xtts-block', v_opt_xtts.get().strip()]
        cmd += ['--text', v_opt_text.get()]
        if v_opt_seeds.get().strip():
            cmd += ['--seeds', v_opt_seeds.get().strip()]
        cmd += ['--budget', v_opt_budget.get().strip() or '25']
        cmd += ['--w-accent', v_opt_wacc.get().strip() or '0.6']
        cmd += ['--w-identity', v_opt_wid.get().strip() or '0.4']
        cmd += ['--max-ref-len', v_opt_maxref.get().strip() or '30']
        cmd += ['--whisper-model', v_opt_model.get()]
        cmd += ['--device', v_opt_device.get()]

        def _opt_on_line(txt):
            s = txt.strip()
            if s.startswith('{1,') or s.startswith('{1 '):
                HANDOFF['win_xtts'] = s
        run_cmd(cmd, console, btn, stop_btn, line_callback=_opt_on_line)

    make_btn(f, ">  Optimise", lancer, 7)

    frm_oho = tk.Frame(f)
    frm_oho.grid(row=9, column=0, columnspan=3, sticky='w', padx=6, pady=(0, 4))
    tk.Label(frm_oho, text="Send winning {} →", fg="gray", font=("Arial", 8)).pack(side='left', padx=(0, 4))
    def _opt_to_cmp():
        block = HANDOFF.get('win_xtts') or v_opt_xtts.get().strip()
        ok = _handoff_set('cmp_xtts', block)
        log(console, "[→] Sent winning {} to Comparator" if ok else "[!] Run the optimiser first.")
    tk.Button(frm_oho, text="→ Comparator", command=_opt_to_cmp).pack(side='left', padx=2)


# ── Tab: Comparator ──────────────────────────────────────────────────────────


# ── Tab: RVC (timbre conversion) ─────────────────────────────────────────────

def tab_rvc(nb):
    f = ttk.Frame(nb)
    nb.add(f, text="[RVC] Timbre")
    f.grid_columnconfigure(1, weight=1)

    # ── 1. Dataset (train data for Applio) ───────────────────────────────────
    tk.Label(f, text="1. Build training dataset (10+ min of the voice)",
             font=("Arial", 9, "bold"), anchor='w').grid(row=0, column=0, columnspan=3,
                                                         sticky='w', padx=6, pady=(6, 2))
    v_rvc_refs = tk.StringVar()
    v_rvc_ds   = tk.StringVar()
    v_rvc_min  = tk.StringVar(value='12')
    try:
        import torch as _t_rvc
        _rvc_dev = 'cuda' if _t_rvc.cuda.is_available() else 'cpu'
    except Exception:
        _rvc_dev = 'cpu'
    v_rvc_dev  = tk.StringVar(value=_rvc_dev)

    add_row(f, "Raw voice ref(s)", v_rvc_refs, 1,
            [("Audio", "*.wav *.mp3 *.flac *.ogg"), ("All", "*.*")],
            multi=True, initialdir=DIR_VOICES)
    add_row(f, "Dataset folder", v_rvc_ds, 2, save=True, initialdir=XTTS_ROOT)

    def _rvc_suggest(*_):
        if v_rvc_refs.get().strip() and not v_rvc_ds.get().strip():
            base = os.path.splitext(os.path.basename(v_rvc_refs.get().split()[0]))[0]
            v_rvc_ds.set(os.path.join(XTTS_ROOT, 'RVC_datasets', base))
    v_rvc_refs.trace_add('write', _rvc_suggest)

    frm_r1 = tk.Frame(f); frm_r1.grid(row=3, column=0, columnspan=3, sticky='w', padx=6, pady=2)
    tk.Label(frm_r1, text="Keep minutes").pack(side='left', padx=(6, 2))
    tk.Entry(frm_r1, textvariable=v_rvc_min, width=5).pack(side='left')
    tk.Label(frm_r1, text="Device").pack(side='left', padx=(10, 2))
    ttk.Combobox(frm_r1, textvariable=v_rvc_dev, values=['cpu', 'cuda'],
                 width=6, state='readonly').pack(side='left')
    tk.Label(frm_r1, text="→ then TRAIN in Applio (docs/RVC_GUIDE.md): 40k, rmvpe, "
                          "250-300 epochs, save every 50", fg='grey',
             font=("Arial", 8)).pack(side='left', padx=(12, 0))

    # ── 2. Convert (through the trained model) ────────────────────────────────
    tk.Label(f, text="2. Convert XTTS output through the trained RVC model",
             font=("Arial", 9, "bold"), anchor='w').grid(row=5, column=0, columnspan=3,
                                                         sticky='w', padx=6, pady=(10, 2))
    v_rvc_applio = tk.StringVar(value=os.path.expanduser('~/Applio'))
    v_rvc_pth    = tk.StringVar()
    v_rvc_idx    = tk.StringVar()
    v_rvc_in     = tk.StringVar()
    v_rvc_out    = tk.StringVar()
    v_rvc_pitch  = tk.StringVar(value='0')
    v_rvc_irate  = tk.StringVar(value='0.75')
    v_rvc_prot   = tk.StringVar(value='0.33')

    add_row(f, "Applio folder", v_rvc_applio, 6)
    add_row(f, "Model (.pth)", v_rvc_pth, 7, [("RVC model", "*.pth"), ("All", "*.*")],
            initialdir=os.path.expanduser('~/Applio/logs'))
    add_row(f, "Index (.index)", v_rvc_idx, 8, [("Index", "*.index"), ("All", "*.*")],
            initialdir=os.path.expanduser('~/Applio/logs'))
    add_row(f, "Input audio", v_rvc_in, 9,
            [("Audio", "*.wav *.mp3"), ("All", "*.*")], initialdir=DIR_OUTPUT)
    add_row(f, "Output audio", v_rvc_out, 10, [("WAV", "*.wav")], save=True,
            initialdir=DIR_OUTPUT)

    def _rvc_out_suggest(*_):
        if v_rvc_in.get().strip() and not v_rvc_out.get().strip():
            v_rvc_out.set(os.path.splitext(v_rvc_in.get().strip())[0] + '_rvc.wav')
    v_rvc_in.trace_add('write', _rvc_out_suggest)

    frm_r2 = tk.Frame(f); frm_r2.grid(row=11, column=0, columnspan=3, sticky='w', padx=6, pady=2)
    for lbl, var, w in [("Pitch (st)", v_rvc_pitch, 4), ("Index rate", v_rvc_irate, 5),
                        ("Protect", v_rvc_prot, 5)]:
        tk.Label(frm_r2, text=lbl).pack(side='left', padx=(6, 2))
        tk.Entry(frm_r2, textvariable=var, width=w).pack(side='left')
    tk.Label(frm_r2, text="(index rate ↑ = more target timbre, more artefacts)",
             fg='grey', font=("Arial", 8)).pack(side='left', padx=(8, 0))

    # ── 3. Measure ────────────────────────────────────────────────────────────
    tk.Label(f, text="3. Measure identity vs the real reference",
             font=("Arial", 9, "bold"), anchor='w').grid(row=12, column=0, columnspan=3,
                                                         sticky='w', padx=6, pady=(10, 2))
    v_rvc_measref = tk.StringVar()
    add_row(f, "Real reference", v_rvc_measref, 13,
            [("Audio", "*.wav *.mp3"), ("All", "*.*")], initialdir=DIR_VOICES)

    console = add_console(f, 15)

    def lancer_dataset(btn, stop_btn=None):
        refs = [r for r in v_rvc_refs.get().split() if r.strip()]
        if not refs:
            log(console, "[ERR] Raw voice reference(s) required."); return
        if not v_rvc_ds.get().strip():
            log(console, "[ERR] Dataset folder required."); return
        cmd = [sys.executable, os.path.join(SCRIPTS_DIR, 'prepare_rvc_dataset.py')]
        cmd += refs + ['-o', v_rvc_ds.get().strip(),
                       '--keep-minutes', v_rvc_min.get().strip() or '12',
                       '--device', v_rvc_dev.get()]
        run_cmd(cmd, console, btn, stop_btn)

    def lancer_convert(btn, stop_btn=None):
        if not (v_rvc_in.get().strip() and v_rvc_pth.get().strip() and v_rvc_idx.get().strip()):
            log(console, "[ERR] Input, model .pth and .index are required."); return
        cmd = [sys.executable, os.path.join(SCRIPTS_DIR, 'rvc_convert.py'),
               v_rvc_in.get().strip(),
               '-o', v_rvc_out.get().strip() or
                     (os.path.splitext(v_rvc_in.get().strip())[0] + '_rvc.wav'),
               '--model', v_rvc_pth.get().strip(),
               '--index', v_rvc_idx.get().strip(),
               '--applio-dir', v_rvc_applio.get().strip() or os.path.expanduser('~/Applio'),
               '--pitch', v_rvc_pitch.get().strip() or '0',
               '--index-rate', v_rvc_irate.get().strip() or '0.75',
               '--protect', v_rvc_prot.get().strip() or '0.33']
        run_cmd(cmd, console, btn, stop_btn)

    def lancer_measure(btn, stop_btn=None):
        ref = v_rvc_measref.get().strip()
        if not ref:
            log(console, "[ERR] Real reference required."); return
        cands = [c for c in [v_rvc_in.get().strip(), v_rvc_out.get().strip()]
                 if c and os.path.exists(c)]
        if not cands:
            log(console, "[ERR] No existing input/converted file to measure."); return
        cmd = [sys.executable, os.path.join(SCRIPTS_DIR, 'speaker_identity.py'), ref] + cands
        run_cmd(cmd, console, btn, stop_btn)

    make_btn(f, ">  1. Build dataset", lancer_dataset, 4)
    make_btn(f, ">  2. Convert", lancer_convert, 14)
    make_btn(f, ">  3. Measure identity (before/after)", lancer_measure, 16)


def tab_comparator(nb):
    f = tk.Frame(nb)
    nb.add(f, text="[Cmp] Comparator")
    f.grid_columnconfigure(1, weight=1)

    v_cmp_ref      = tk.StringVar()
    v_cmp_lang     = tk.StringVar(value='FR')
    v_cmp_xtts     = tk.StringVar()
    v_cmp_audio    = tk.StringVar()
    v_cmp_text     = tk.StringVar(value="Bonjour, voici un court test de comparaison de voix.")
    v_cmp_output   = tk.StringVar()
    v_cmp_opt_out  = tk.StringVar()
    TARGETS['cmp_xtts']  = v_cmp_xtts
    TARGETS['cmp_audio'] = v_cmp_audio

    row = 0
    # Reference file — used for both comparison AND XTTS cloning
    tk.Label(f, text="Reference voice", anchor="w", width=18).grid(row=row, column=0, sticky="w", padx=6, pady=3)
    tk.Label(f, text="used for comparison AND as XTTS voice reference",
             fg="gray", font=("Arial",8)).grid(row=row, column=1, sticky="w", padx=4)
    row += 1
    frm_r = tk.Frame(f); frm_r.grid(row=row, column=1, sticky="ew", padx=4)
    frm_r.grid_columnconfigure(0, weight=1)
    tk.Entry(frm_r, textvariable=v_cmp_ref).grid(row=0, column=0, sticky="ew")
    tk.Button(frm_r, text="Browse", width=8,
        command=lambda: browse_file(v_cmp_ref, [("Audio","*.wav *.mp3 *.flac"),("All","*.*")], initialdir=DIR_VOICES)
    ).grid(row=0, column=1, padx=2)

    row += 1
    tk.Label(f, text="Language", anchor="w", width=18).grid(row=row, column=0, sticky="w", padx=6, pady=3)
    ttk.Combobox(f, textvariable=v_cmp_lang, values=LANGS, width=8, state="readonly").grid(row=row, column=1, sticky="w", padx=4)

    row += 1
    tk.Label(f, text="XTTS params {}", anchor="w", width=18).grid(row=row, column=0, sticky="w", padx=6, pady=3)
    tk.Label(f, text="{N, seed, trim_start, trim_end, fade_in, fade_out, temp, top_k, top_p, rep_pen, len_pen, gpt_cond_len, gpt_cond_chunk_len, sound_norm_refs}",
             fg="gray", font=("Arial",8)).grid(row=row, column=1, sticky="w", padx=4)
    row += 1
    tk.Entry(f, textvariable=v_cmp_xtts).grid(row=row, column=1, sticky="ew", padx=4)

    row += 1
    tk.Label(f, text="Audio params []", anchor="w", width=18).grid(row=row, column=0, sticky="w", padx=6, pady=3)
    tk.Label(f, text="[N, LANG, speed, vol, eq_low, eq_mid, eq_high, hp, lp, NR, comp, de-ess, reverb, noise_gate, pan, limiter]",
             fg="gray", font=("Arial",8)).grid(row=row, column=1, sticky="w", padx=4)
    row += 1
    tk.Entry(f, textvariable=v_cmp_audio).grid(row=row, column=1, sticky="ew", padx=4)

    row += 1
    tk.Label(f, text="Test text", anchor="w", width=18).grid(row=row, column=0, sticky="nw", padx=6, pady=3)
    tk.Label(f, text="supports [pause=1s] syntax", fg="gray", font=("Arial",8)).grid(row=row, column=1, sticky="w", padx=4)
    row += 1
    cmp_editor_frame = tk.Frame(f)
    cmp_editor_frame.grid(row=row, column=0, columnspan=2, sticky="ew", padx=6, pady=2)
    cmp_editor_frame.grid_columnconfigure(0, weight=1)
    cmp_editor = scrolledtext.ScrolledText(cmp_editor_frame, height=5, font=("Courier", 9), wrap="word", undo=True)
    cmp_editor.pack(fill="both", expand=True)
    cmp_editor.insert("1.0", "Bonjour, voici un court test de comparaison de voix.\n[pause=1s]\nLa voix doit être naturelle et fluide.")

    row += 1
    tk.Label(f, text="Clone output", anchor="w", width=18).grid(row=row, column=0, sticky="w", padx=6, pady=3)
    frm_o = tk.Frame(f); frm_o.grid(row=row, column=1, sticky="ew", padx=4)
    frm_o.grid_columnconfigure(0, weight=1)
    tk.Entry(frm_o, textvariable=v_cmp_output).grid(row=0, column=0, sticky="ew")
    tk.Button(frm_o, text="Browse", width=8,
        command=lambda: browse_file(v_cmp_output, [("WAV","*.wav")], save=True, initialdir=DIR_OUTPUT)
    ).grid(row=0, column=1, padx=2)

    row += 1
    tk.Label(f, text="Optimised output", anchor="w", width=18).grid(row=row, column=0, sticky="w", padx=6, pady=3)
    frm_p = tk.Frame(f); frm_p.grid(row=row, column=1, sticky="ew", padx=4)
    frm_p.grid_columnconfigure(0, weight=1)
    tk.Entry(frm_p, textvariable=v_cmp_opt_out).grid(row=0, column=0, sticky="ew")
    tk.Button(frm_p, text="Browse", width=8,
        command=lambda: browse_file(v_cmp_opt_out, [("WAV","*.wav")], save=True, initialdir=DIR_OUTPUT)
    ).grid(row=0, column=1, padx=2)

    row += 1
    frm_iter = tk.Frame(f)
    frm_iter.grid(row=row, column=0, columnspan=2, sticky='w', padx=6, pady=4)
    tk.Label(frm_iter, text="Iterations", width=10, anchor='w').pack(side='left')
    v_cmp_iter = tk.StringVar(value='1')
    tk.Spinbox(frm_iter, from_=1, to=20, textvariable=v_cmp_iter, width=4).pack(side='left', padx=4)
    tk.Label(frm_iter, text="  Stop if score Δ <", anchor='w').pack(side='left', padx=(12,2))
    v_cmp_conv = tk.StringVar(value='0.5')
    tk.Entry(frm_iter, textvariable=v_cmp_conv, width=5).pack(side='left')
    tk.Label(frm_iter, text="or [] unchanged", fg='gray', font=('Arial',8)).pack(side='left', padx=6)

    row += 1
    console = add_console(f, row)

    def lancer(btn, stop_btn=None):
        ref = v_cmp_ref.get().strip()
        if not ref:
            log(console, "[ERR] Reference voice file required."); return
        if not v_cmp_xtts.get().strip() or not v_cmp_audio.get().strip():
            log(console, "[ERR] XTTS and Audio blocks required."); return

        output = v_cmp_output.get().strip() or os.path.join(DIR_OUTPUT, "comparator_clone.wav")
        opt_output = v_cmp_opt_out.get().strip() or os.path.join(DIR_OUTPUT, "comparator_optimised.wav")
        v_cmp_output.set(output)
        v_cmp_opt_out.set(opt_output)

        refs = ref.split()
        n_iter = int(v_cmp_iter.get() or 1)
        conv_thr = float(v_cmp_conv.get() or 0.5)

        # Write text to temp file
        import tempfile as _tf
        _tfile = _tf.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8')
        _tfile.write(cmp_editor.get("1.0", "end-1c"))
        _tfile.close()

        def build_cmd(audio_block_str, out_clone, out_opt, max_iter, conv):
            cmd = [sys.executable, os.path.join(SCRIPTS_DIR, "voice_comparator.py")]
            cmd += [ref]
            cmd += refs
            cmd += [v_cmp_lang.get()]
            cmd += ["--xtts-block", v_cmp_xtts.get().strip()]
            cmd += ["--audio-block", audio_block_str]
            cmd += ["--text-file", _tfile.name]
            cmd += ["--output", out_clone]
            cmd += ["--output-optimised", out_opt]
            cmd += ["--iterations", str(max_iter)]
            cmd += ["--conv-threshold", str(conv)]
            return cmd

        def _on_line(txt):
            # Update Audio params [] field when Next [] is detected
            if 'Next []' in txt and '[1,' in txt:
                # Extract the data block [1, FR, ...] — last [ in the line
                idx = txt.rfind('[1,')
                if idx >= 0:
                    v_cmp_audio.set(txt[idx:])

        cmd = build_cmd(v_cmp_audio.get().strip(), output, opt_output, n_iter, conv_thr)
        run_cmd(cmd, console, btn, stop_btn, line_callback=_on_line)

    make_btn(f, ">  Compare & Optimise", lancer, row + 1)

# ── Main ───────────────────────────────────────────────────────────────────

def main():
    global _root
    root = tk.Tk()
    _root = root
    root.title("XTTS Voice Studio")
    root.geometry("800x780")
    root.resizable(True, True)

    style = ttk.Style()
    style.theme_use('clam')
    style.configure('TNotebook.Tab', font=('Arial', 10), padding=[10, 4])

    header = tk.Frame(root, bg='#1a1a2e', height=50)
    header.pack(fill='x')
    header.pack_propagate(False)
    tk.Label(header, text="XTTS Voice Studio",
             bg='#1a1a2e', fg='white',
             font=('Arial', 14, 'bold')).pack(pady=10)

    # Global player bar
    player_bar = tk.Frame(root, bg='#2c2c2c', pady=4)
    player_bar.pack(fill='x', padx=8, pady=(4,0))
    tk.Label(player_bar, text="Player:", bg='#2c2c2c', fg='white',
             font=('Arial',9,'bold'), width=7).pack(side='left', padx=4)
    v_player_path = tk.StringVar()
    tk.Entry(player_bar, textvariable=v_player_path, width=50).pack(side='left', padx=4, fill='x', expand=True)
    def _pbrowse():
        p = filedialog.askopenfilename(filetypes=[("Audio","*.wav *.mp3 *.flac *.ogg"),("All","*.*")], initialdir=DIR_MP3)
        if p: v_player_path.set(p)
    tk.Button(player_bar, text="Browse", width=8, command=_pbrowse).pack(side='left', padx=2)
    play_btn = tk.Button(player_bar, text="> Play", width=8, bg='#1a6b9e', fg='white', font=('Arial',9,'bold'))
    play_btn.config(command=lambda b=play_btn: play_toggle(v_player_path.get(), b))
    play_btn.pack(side='left', padx=4)
    nb = ttk.Notebook(root)
    nb.pack(fill='both', expand=True, padx=8, pady=8)

    tab_generator(nb)
    tab_auto(nb)
    tab_curate(nb)
    tab_analyser(nb)
    tab_transcribe(nb)
    tab_extract(nb)
    tab_pitch(nb)
    tab_convert(nb)
    tab_validator(nb)
    tab_optimize(nb)
    tab_comparator(nb)
    tab_rvc(nb)

    def on_close():
        _stop_player()       # stop audio player if running
        root.destroy()
        os._exit(0)          # force-kill any background threads/processes

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
