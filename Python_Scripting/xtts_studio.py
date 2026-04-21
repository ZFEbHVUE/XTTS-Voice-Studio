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


# Default directories

XTTS_ROOT     = os.path.expanduser("~/XTTS-Voice-Studio")
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

def run_cmd(cmd, console, btn, stop_btn=None):
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
                text=True, bufsize=1, env=env)
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
                if txt: log(console, txt)
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
                pgid = _os.getpgid(proc_holder[0].pid)
                _os.killpg(pgid, _sig.SIGTERM)
                _os.killpg(pgid, _sig.SIGKILL)   # immediately force-kill — Python ignores SIGTERM
            except Exception:
                try:
                    proc_holder[0].kill()          # SIGKILL directly
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
    v_voices  = tk.StringVar()
    v_ambient = tk.StringVar()
    v_music   = tk.StringVar()

    add_row(f, "Script (.txt)",  v_script,  0, [("Text","*.txt"),("All","*.*")], initialdir=DIR_PROMPTS)
    add_row(f, "Output (wav/mp3)", v_output, 1, [("WAV","*.wav"),("MP3","*.mp3"),("FLAC","*.flac"),("OGG","*.ogg"),("All","*.*")], save=True, initialdir=DIR_OUTPUT)
    add_row(f, "Voices (1+)",     v_voices,  2, [("Audio","*.wav *.mp3 *.flac *.ogg"),("All","*.*")], multi=True, initialdir=DIR_VOICES)
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
    editor.config(yscrollcommand=scroll_e.set)
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
        if not v_script.get() or not v_output.get() or not v_voices.get():
            log(console, "ERR Script, sortie et voix obligatoires."); return
        cmd = [sys.executable,
               os.path.join(SCRIPTS_DIR, 'guided_meditation_generator_v23.py'),
               v_script.get(), v_output.get()]
        cmd += v_voices.get().split()
        if v_ambient.get(): cmd += v_ambient.get().split()
        if v_music.get():   cmd += v_music.get().split()
        cmd += ['--mp3-bitrate', v_gen_mp3_bitrate.get(), '--mp3-mode', v_gen_mp3_mode.get()]
        run_cmd(cmd, console, btn, stop_btn)

    make_btn(f, ">  Run", lancer, 9)


# ── Tab: Analyser ───────────────────────────────────────────────────────────

def tab_analyser(nb):
    f = ttk.Frame(nb)
    nb.add(f, text="[Ana] Analyser")
    f.grid_columnconfigure(0, weight=1)
    f.grid_rowconfigure(2, weight=1)

    LANGS = ['FR','EN','ES','DE','IT','PT','PL','TR','RU','NL','CS','AR','ZH-CN','HU','KO','JA','HI']
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
        v_precise = tk.BooleanVar(value=False)

        row_f = tk.Frame(voices_frame)
        row_f.pack(fill='x', padx=4, pady=2)

        tk.Label(row_f, text="V", width=2).pack(side='left')
        tk.Spinbox(row_f, from_=1, to=20, textvariable=v_num, width=3).pack(side='left', padx=1)
        tk.Entry(row_f, textvariable=v_path).pack(side='left', fill='x', expand=True, padx=3)
        tk.Button(row_f, text="Browse", width=8,
            command=lambda vp=v_path: browse_file(vp, [("Audio","*.wav *.mp3 *.flac *.ogg"),("All","*.*")], initialdir=DIR_VOICES)).pack(side='left', padx=1)

        ttk.Combobox(row_f, textvariable=v_lang, values=LANGS,
                     width=6, state='readonly').pack(side='left', padx=1)
        tk.Label(row_f, text="Seed:").pack(side='left', padx=(4,0))
        tk.Spinbox(row_f, from_=0, to=99999, textvariable=v_seed, width=6).pack(side='left', padx=1)
        tk.Checkbutton(row_f, text="Prec", variable=v_precise).pack(side='left', padx=1)

        entry = (v_path, v_lang, v_num, v_seed, v_precise, row_f)

        def remove(e=entry):
            e[5].destroy()
            voice_rows.remove(e)

        tk.Button(row_f, text="X", width=2, fg='red', command=remove).pack(side='left', padx=1)
        voice_rows.append(entry)

    add_voice_row(1)
    add_voice_row(2)

    # Add button + precise option
    ctrl_frame = tk.Frame(f)
    ctrl_frame.grid(row=1, column=0, columnspan=3, sticky='w', padx=6, pady=2)
    tk.Button(ctrl_frame, text="+ Add voice",
        command=lambda: add_voice_row(),
        bg='#444', fg='white', width=14).pack(side='left', padx=4)
    tk.Label(ctrl_frame, text="Prec = precise mode",
             fg='gray', font=('Arial',8)).pack(side='left', padx=6)
    v_f0_engine = tk.StringVar(value='auto')
    ttk.Combobox(ctrl_frame, textvariable=v_f0_engine, width=8, state='readonly',
        values=['auto','crepe','pyin']).pack(side='left', padx=2)
    tk.Label(ctrl_frame, text="F0 engine (auto=crepe if available)",
             fg='gray', font=('Arial',8)).pack(side='left', padx=4)

    console = add_console(f, 2)

    def lancer(btn, stop_btn=None):
        valids = [(vp.get(), vl.get(), vn.get(), vs.get(), vpr.get())
                  for vp, vl, vn, vs, vpr, _ in voice_rows if vp.get().strip()]
        if not valids:
            log(console, "[ERR] Add at least one voice."); return

        # One call per voice (individual precise mode)
        
        cmds = []
        for vpath, vlang, vnum, vseed, vprec in valids:
            cmd = [sys.executable, os.path.join(SCRIPTS_DIR, 'voice_analyser.py')]
            if vprec:
                cmd.append('--precise')
            cmd += ['--f0-engine', v_f0_engine.get()]
            cmd += ['--start-num', str(vnum)]
            if vseed != 0:
                cmd += ['--seed', str(vseed)]
            cmd += [vpath, vlang]
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

    v_input   = tk.StringVar()
    v_output  = tk.StringVar()
    v_keep    = tk.StringVar(value='female')
    v_silence = tk.StringVar(value='auto')
    v_thr     = tk.StringVar(value='165')
    v_deverb  = tk.StringVar(value='none')
    v_debug   = tk.BooleanVar(value=False)

    add_row(f, "Audio source",   v_input,  0, [("Audio","*.wav *.mp3 *.flac *.ogg"),("All","*.*")], initialdir=DIR_VOICES)
    add_row(f, "Output (wav/mp3)", v_output, 1, [("WAV","*.wav"),("MP3","*.mp3"),("FLAC","*.flac"),("OGG","*.ogg"),("All","*.*")], save=True, initialdir=DIR_OUTPUT)

    tk.Label(f, text="Keep", anchor='w', width=20).grid(row=2, column=0, sticky='w', padx=6, pady=3)
    ttk.Combobox(f, textvariable=v_keep, width=14, state='readonly',
        values=['female','male','overlap','all','female,male','vocals only']
    ).grid(row=2, column=1, sticky='w', padx=4)

    tk.Label(f, text="Silence (s/auto/0)", anchor='w', width=20).grid(row=3, column=0, sticky='w', padx=6, pady=3)
    tk.Entry(f, textvariable=v_silence, width=8).grid(row=3, column=1, sticky='w', padx=4)

    tk.Label(f, text="F0 threshold (Hz)", anchor='w', width=20).grid(row=4, column=0, sticky='w', padx=6, pady=3)
    tk.Entry(f, textvariable=v_thr, width=8).grid(row=4, column=1, sticky='w', padx=4)

    v_ovrange   = tk.StringVar(value='200')
    v_minsilence = tk.StringVar(value='0.30')
    tk.Label(f, text="Overlap range (Hz)", anchor='w', width=20).grid(row=5, column=0, sticky='w', padx=6, pady=3)
    tk.Entry(f, textvariable=v_ovrange, width=8).grid(row=5, column=1, sticky='w', padx=4)

    tk.Label(f, text="Min silence (s)", anchor='w', width=20).grid(row=6, column=0, sticky='w', padx=6, pady=3)
    tk.Entry(f, textvariable=v_minsilence, width=8).grid(row=6, column=1, sticky='w', padx=4)

    tk.Label(f, text="Dereverberation", anchor='w', width=20).grid(row=7, column=0, sticky='w', padx=6, pady=3)
    ttk.Combobox(f, textvariable=v_deverb, width=14, state='readonly',
        values=['none','noisereduce','wpe','deepfilter']
    ).grid(row=7, column=1, sticky='w', padx=4)

    tk.Checkbutton(f, text="Debug mode", variable=v_debug).grid(
        row=8, column=1, sticky='w', padx=4, pady=3)

    v_remove_music = tk.BooleanVar(value=False)
    v_demucs_model = tk.StringVar(value='htdemucs_ft')
    v_mp3_bitrate  = tk.StringVar(value='192')
    v_mp3_mode     = tk.StringVar(value='cbr')

    # ── Device selector (default: auto-detect) ───────────────────────────
    try:
        import torch as _torch_vox
        _vox_default = "cuda" if _torch_vox.cuda.is_available() else "cpu"
    except Exception:
        _vox_default = "cpu"
    v_vox_device = tk.StringVar(value=_vox_default)
    tk.Label(f, text="Device", anchor='w', width=20).grid(row=8, column=0, sticky='w', padx=6, pady=3)
    ttk.Combobox(f, textvariable=v_vox_device, width=8, state='readonly',
        values=['cpu', 'cuda']).grid(row=8, column=1, sticky='w', padx=4)

    def on_remove_music_toggle(*args):
        if v_remove_music.get():
            v_keep.set('vocals only')
        else:
            if v_keep.get() == 'vocals only':
                v_keep.set('female')

    v_remove_music.trace_add('write', on_remove_music_toggle)

    tk.Checkbutton(f, text="Remove background music (demucs)",
                   variable=v_remove_music).grid(
        row=9, column=0, columnspan=2, sticky='w', padx=6, pady=2)

    tk.Label(f, text="Demucs model", anchor='w', width=20).grid(
        row=10, column=0, sticky='w', padx=6, pady=3)
    ttk.Combobox(f, textvariable=v_demucs_model, width=16, state='readonly',
        values=['htdemucs', 'htdemucs_ft', 'mdx_extra']
    ).grid(row=10, column=1, sticky='w', padx=4)

    # ── MP3 output options ───────────────────────────────────────────────
    tk.Label(f, text="MP3 bitrate (kbps)", anchor='w', width=20).grid(row=12, column=0, sticky='w', padx=6, pady=3)
    frm_mp3 = tk.Frame(f)
    frm_mp3.grid(row=11, column=1, sticky='w', padx=4)
    ttk.Combobox(frm_mp3, textvariable=v_mp3_bitrate, width=6, state='readonly',
        values=['128','160','192','256','320']).pack(side='left')
    ttk.Combobox(frm_mp3, textvariable=v_mp3_mode, width=5, state='readonly',
        values=['cbr','vbr']).pack(side='left', padx=6)
    tk.Label(frm_mp3, text="(only used if output is .mp3)", fg='grey').pack(side='left')

    console = add_console(f, 14)

    def lancer(btn, stop_btn=None):
        if not v_input.get() or not v_output.get():
            log(console, "[ERR] Source and output required."); return
        cmd = [sys.executable, os.path.join(SCRIPTS_DIR, 'extract_voices.py'),
               v_input.get(), v_output.get(),
               '--keep', v_keep.get(),
               '--silence', v_silence.get(),
               '--threshold', v_thr.get(),
               '--overlap-range', v_ovrange.get(),
               '--min-silence', v_minsilence.get(),
               '--dereverberate', v_deverb.get()]
        if v_debug.get():
            cmd.append('--debug')
        cmd += ['--device', v_vox_device.get()]
        cmd += ['--mp3-bitrate', v_mp3_bitrate.get(), '--mp3-mode', v_mp3_mode.get()]
        if v_remove_music.get():
            cmd += ['--remove-music', '--demucs-model', v_demucs_model.get()]

        run_cmd(cmd, console, btn, stop_btn)

    make_btn(f, ">  Separate", lancer, 13)


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

    # ── MP3 options (only used if output is .mp3) ─────────────────────────
    v_vid_mp3_bitrate = tk.StringVar(value='192')
    v_vid_mp3_mode    = tk.StringVar(value='cbr')
    tk.Label(f, text="MP3 bitrate (kbps)", anchor='w', width=20).grid(row=2, column=0, sticky='w', padx=6, pady=3)
    frm_vid_mp3 = tk.Frame(f)
    frm_vid_mp3.grid(row=2, column=1, sticky='w', padx=4)
    ttk.Combobox(frm_vid_mp3, textvariable=v_vid_mp3_bitrate, width=6, state='readonly',
        values=['128','160','192','256','320']).pack(side='left')
    ttk.Combobox(frm_vid_mp3, textvariable=v_vid_mp3_mode, width=5, state='readonly',
        values=['cbr','vbr']).pack(side='left', padx=6)
    tk.Label(frm_vid_mp3, text="(only used if output is .mp3)", fg='grey').pack(side='left')

    console = add_console(f, 4)

    def lancer(btn, stop_btn=None):
        if not v_input.get() or not v_output.get():
            log(console, "[ERR] Source and output required."); return
        ext = os.path.splitext(v_output.get())[1].lower()
        if ext == '.mp3':
            br  = v_vid_mp3_bitrate.get()
            if v_vid_mp3_mode.get() == 'vbr':
                vbr_map = {'128':'6','160':'5','192':'4','256':'2','320':'0'}
                cmd = ['ffmpeg', '-y', '-i', v_input.get(), '-vn',
                       '-codec:a', 'libmp3lame', '-q:a', vbr_map.get(br,'4'),
                       v_output.get()]
            else:
                cmd = ['ffmpeg', '-y', '-i', v_input.get(), '-vn',
                       '-codec:a', 'libmp3lame', '-b:a', f'{br}k',
                       v_output.get()]
        elif ext == '.flac':
            cmd = ['ffmpeg', '-y', '-i', v_input.get(), '-vn',
                   '-codec:a', 'flac', v_output.get()]
        elif ext == '.ogg':
            cmd = ['ffmpeg', '-y', '-i', v_input.get(), '-vn',
                   '-codec:a', 'libvorbis', v_output.get()]
        else:  # .wav or anything else
            cmd = ['ffmpeg', '-y', '-i', v_input.get(), '-vn', v_output.get()]
        run_cmd(cmd, console, btn, stop_btn)

    make_btn(f, ">  Convert", lancer, 3)


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    global _root
    root = tk.Tk()
    _root = root
    root.title("XTTS Voice Studio")
    root.geometry("800x640")
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
    tab_analyser(nb)
    tab_transcribe(nb)
    tab_extract(nb)
    tab_pitch(nb)
    tab_convert(nb)

    root.mainloop()


if __name__ == "__main__":
    main()
