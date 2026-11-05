"""XTTS Studio file selection using only Tk/ttk, in the application's theme.

Filesystem scans and optional audio probes run off the Tk thread. Workers
never hold widgets or call Tk. Only the optional audio player is a subprocess.
"""
from dataclasses import dataclass
from datetime import datetime
import fnmatch
import json
import math
import os
from pathlib import Path
import queue
import shutil
import subprocess
import tempfile
import threading
import wave
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog

AUDIO_EXTENSIONS = frozenset(('.wav', '.mp3', '.flac', '.ogg', '.opus', '.m4a',
                              '.aac', '.aif', '.aiff', '.wma', '.mp4'))
_PARENT = lambda: tk._default_root
_SHORTCUTS = lambda: []
_BEFORE_PREVIEW = lambda: None
_ACTIVE = None
_PREFS = None


def configure(parent=None, shortcuts=None, before_preview=None):
    """Supply lazy callbacks; importing this module does not create a Tk root."""
    global _PARENT, _SHORTCUTS, _BEFORE_PREVIEW
    if parent is not None:
        _PARENT = parent
    if shortcuts is not None:
        _SHORTCUTS = shortcuts
    if before_preview is not None:
        _BEFORE_PREVIEW = before_preview


def _preferences_path():
    base = os.environ.get('XDG_CONFIG_HOME')
    if not base:
        base = os.environ.get('APPDATA') if os.name == 'nt' else str(Path.home() / '.config')
    return Path(base or Path.home()) / 'xtts_studio' / 'file_dialog.json'


def _preferences():
    global _PREFS
    if _PREFS is None:
        try:
            value = json.loads(_preferences_path().read_text(encoding='utf-8'))
            _PREFS = value if isinstance(value, dict) else {}
        except (OSError, ValueError):
            _PREFS = {}
    return _PREFS


def _save_preferences(value):
    temporary = None
    try:
        path = _preferences_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=path.parent,
                                         prefix='.file_dialog_', delete=False) as handle:
            temporary = handle.name
            json.dump(value, handle, ensure_ascii=True, indent=2)
        os.replace(temporary, path)
        temporary = None
    except (OSError, ValueError, TypeError):
        pass  # An unwritable preferences directory never prevents file selection.
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except OSError:
                pass


@dataclass(frozen=True)
class FileEntry:
    path: str
    name: str
    is_dir: bool
    size: int
    modified: float

    @property
    def kind(self):
        return 'Dossier' if self.is_dir else (Path(self.name).suffix.lstrip('.').upper() or 'Fichier')


def _scan_directory(path, token, stopped, results):
    try:
        entries = []
        with os.scandir(path) as items:
            for item in items:
                if stopped.is_set():
                    return
                try:
                    info = item.stat()
                    entries.append(FileEntry(item.path, item.name, item.is_dir(),
                                             info.st_size, info.st_mtime))
                except OSError:
                    continue  # A file may disappear during a directory scan.
        if not stopped.is_set():
            results.put(('scan', token, path, entries, None))
    except OSError as exc:
        if not stopped.is_set():
            results.put(('scan', token, path, [], str(exc)))


def _filetypes(value):
    result = []
    for label, patterns in value or [('Tous les fichiers', '*')]:
        if isinstance(patterns, str):
            patterns = patterns.split()
        patterns = tuple('*' if str(p) == '*.*' else str(p) for p in patterns)
        if patterns:
            result.append((str(label), patterns))
    return result or [('Tous les fichiers', ('*',))]


def _visible_entries(entries, patterns, search='', hidden=False, column='name', reverse=False):
    needle = search.casefold()
    visible = [e for e in entries if (hidden or not e.name.startswith('.'))
               and needle in e.name.casefold()
               and (e.is_dir or any(fnmatch.fnmatchcase(e.name.casefold(), p.casefold()) for p in patterns))]
    def key(entry):
        value = {'name': entry.name.casefold(), 'kind': entry.kind,
                 'size': entry.size, 'modified': entry.modified}[column]
        return value, entry.name.casefold()
    # Keep folders first, including when a numeric/date column is descending.
    return (sorted((e for e in visible if e.is_dir), key=key, reverse=reverse)
            + sorted((e for e in visible if not e.is_dir), key=key, reverse=reverse))


def _size_label(size):
    if size < 1024:
        return f'{size} o'
    for unit in ('Kio', 'Mio', 'Gio', 'Tio'):
        size /= 1024
        if size < 1024 or unit == 'Tio':
            return f'{size:.1f} {unit}'


def _with_extension(path, extension):
    if extension and not os.path.splitext(path)[1]:
        return path + (extension if extension.startswith('.') else '.' + extension)
    return path


def _audio_info(path):
    """Read headers only. Compressed formats use the app's optional ffprobe."""
    if Path(path).suffix.lower() not in AUDIO_EXTENSIONS:
        return None
    if Path(path).suffix.lower() == '.wav':
        try:
            with wave.open(str(path), 'rb') as audio:
                return {'duration': audio.getnframes() / audio.getframerate(),
                        'rate': audio.getframerate(), 'channels': audio.getnchannels()}
        except (OSError, wave.Error, EOFError, ZeroDivisionError):
            pass
    ffprobe = shutil.which('ffprobe')
    if not ffprobe:
        return None
    flags = {'creationflags': getattr(subprocess, 'CREATE_NO_WINDOW', 0)} if os.name == 'nt' else {}
    try:
        result = subprocess.run([ffprobe, '-v', 'error', '-select_streams', 'a:0',
                                 '-show_entries', 'format=duration:stream=sample_rate,channels,duration',
                                 '-of', 'json', str(path)], capture_output=True, text=True,
                                timeout=4, **flags)
        if result.returncode:
            return None
        data = json.loads(result.stdout)
        stream = data.get('streams', [{}])[0]
        duration = 0
        for raw in (stream.get('duration'), data.get('format', {}).get('duration')):
            try:
                candidate = float(raw)
            except (TypeError, ValueError):
                continue
            if math.isfinite(candidate) and candidate >= 0:
                duration = candidate
                break
        return {'duration': duration,
                'rate': int(stream.get('sample_rate', 0)),
                'channels': int(stream.get('channels', 0))}
    except (OSError, subprocess.SubprocessError, ValueError, IndexError, TypeError):
        return None


def _probe_worker(path, token, results):
    results.put(('audio', token, path, _audio_info(path)))


class PreviewPlayer:
    """A bounded, independent preview. No Tk access from player cleanup."""
    def __init__(self):
        self.process = self.error_file = None

    def start(self, path):
        self.stop()
        ffplay = shutil.which('ffplay')
        if ffplay:
            command = [ffplay, '-nodisp', '-autoexit', '-loglevel', 'error', '-i', str(path)]
        elif Path(path).suffix.lower() == '.wav' and shutil.which('aplay'):
            command = [shutil.which('aplay'), '-q', str(path)]
        else:
            raise OSError('Préécoute indisponible : ffplay est nécessaire pour ce format.')
        error_file = tempfile.TemporaryFile()
        flags = {'creationflags': getattr(subprocess, 'CREATE_NO_WINDOW', 0)} if os.name == 'nt' else {}
        try:
            self.process = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                                            stdout=subprocess.DEVNULL, stderr=error_file, **flags)
            self.error_file = error_file
        except OSError:
            error_file.close()
            raise

    def poll(self):
        if self.process is None or self.process.poll() is None:
            return None
        code = self.process.returncode
        self.process = None
        self.error_file.close()
        self.error_file = None
        return code

    def stop(self):
        proc, error_file = self.process, self.error_file
        self.process = self.error_file = None
        if proc is None:
            return
        try:
            proc.terminate()
        except OSError:
            pass
        def reap():
            try:
                proc.wait(timeout=.3)
            except subprocess.TimeoutExpired:
                try:
                    proc.kill()
                    proc.wait(timeout=1)
                except (OSError, subprocess.TimeoutExpired):
                    pass
            finally:
                if error_file is not None:
                    error_file.close()
        threading.Thread(target=reap, daemon=True).start()


class StudioFileDialog(tk.Toplevel):
    """One child Tk window; the parent is never withdrawn or lowered."""
    def __init__(self, parent, mode='open', **options):
        super().__init__(parent)
        self.owner = parent.winfo_toplevel()
        self.mode, self.options = mode, options
        self.result = () if mode == 'multiple' else ''
        self._closed = False
        self._jobs = set()
        self._results = queue.Queue()
        self._scan_stop = threading.Event()
        self._scan_token = self._audio_token = self._render_token = 0
        self._metadata_worker = None
        self._entries, self._rows = [], {}
        self._history, self._history_index = [], -1
        self._sort_column, self._sort_reverse = 'name', False
        self._loading = False
        self._selected_text = ''
        self.current_dir = str(Path.home())
        self.player = PreviewPlayer()
        self.preferences = _preferences()
        self.filters = _filetypes(options.get('filetypes'))
        self._filter_labels = [f'{label} ({" ".join(patterns)})' for label, patterns in self.filters]
        self.path_var = tk.StringVar(master=self)
        self.name_var = tk.StringVar(master=self, value=options.get('initialfile', ''))
        self.search_var = tk.StringVar(master=self)
        self.filter_var = tk.StringVar(master=self, value=self._filter_labels[0])
        self.hidden_var = tk.BooleanVar(master=self, value=bool(self.preferences.get('hidden', False)))
        self.status_var = tk.StringVar(master=self)
        self.details_var = tk.StringVar(master=self, value='Sélectionnez un fichier pour afficher ses informations.')
        self.title(options.get('title') or {'save': 'Enregistrer sous', 'directory': 'Choisir un dossier'}.get(mode, 'Ouvrir un fichier'))
        if self.owner.winfo_viewable():
            self.transient(self.owner)
        self.protocol('WM_DELETE_WINDOW', self.cancel)
        self.bind('<Destroy>', self._destroyed, add='+')
        self._build()
        self.bind('<Escape>', lambda event: self.cancel())
        self.bind('<Return>', lambda event: self.accept())
        self.bind('<Alt-Up>', lambda event: self._up())
        self.bind('<Alt-Left>', lambda event: self._history_move(-1))
        self.bind('<Alt-Right>', lambda event: self._history_move(1))
        self.bind('<Control-l>', self._focus_address)
        self.bind('<Control-h>', self._toggle_hidden)
        self.bind('<F5>', lambda event: self.navigate(self.current_dir, remember=False))
        self.search_var.trace_add('write', lambda *unused: self._render())
        initial = options.get('initialdir')
        if not initial:
            saved = self.preferences.get('last_dirs', {})
            initial = saved.get(self._preference_key()) if isinstance(saved, dict) else None
        initial = os.path.abspath(os.path.expanduser(str(initial or os.getcwd())))
        if not os.path.isdir(initial):
            initial = str(Path.home())
        self._position()
        self.navigate(initial)
        self._later(40, self._drain)

    def _later(self, delay, callback):
        def run():
            self._jobs.discard(job)
            if not self._closed:
                callback()
        job = self.after(delay, run)
        self._jobs.add(job)
        return job

    def _build(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)
        toolbar = ttk.Frame(self, padding=(10, 10, 10, 4))
        toolbar.grid(row=0, column=0, sticky='ew')
        toolbar.columnconfigure(3, weight=1)
        self.back_button = ttk.Button(toolbar, text='Préc.', width=6, command=lambda: self._history_move(-1))
        self.back_button.grid(row=0, column=0, padx=(0, 3))
        self.forward_button = ttk.Button(toolbar, text='Suiv.', width=6, command=lambda: self._history_move(1))
        self.forward_button.grid(row=0, column=1, padx=3)
        ttk.Button(toolbar, text='Parent', width=6, command=self._up).grid(row=0, column=2, padx=(3, 6))
        self.address = ttk.Entry(toolbar, textvariable=self.path_var)
        self.address.grid(row=0, column=3, sticky='ew')
        self.address.bind('<Return>', self._address_enter)
        ttk.Button(toolbar, text='Actualiser', command=lambda: self.navigate(self.current_dir, remember=False)).grid(row=0, column=4, padx=(6, 0))
        tools = ttk.Frame(self, padding=(10, 2, 10, 6))
        tools.grid(row=1, column=0, sticky='ew')
        tools.columnconfigure(1, weight=1)
        ttk.Label(tools, text='Rechercher :').grid(row=0, column=0, padx=(0, 6))
        search = ttk.Entry(tools, textvariable=self.search_var)
        search.grid(row=0, column=1, sticky='ew')
        search.bind('<Return>', lambda event: 'break')
        ttk.Checkbutton(tools, text='Fichiers cachés', variable=self.hidden_var, command=self._render).grid(row=0, column=2, padx=10)
        ttk.Button(tools, text='Nouveau dossier', command=self._new_folder).grid(row=0, column=3)
        browser = ttk.Frame(self, padding=(10, 0))
        browser.grid(row=2, column=0, sticky='nsew')
        browser.columnconfigure(1, weight=1)
        browser.rowconfigure(0, weight=1)
        self.places = ttk.Frame(browser, padding=(0, 0, 10, 0))
        self.places.grid(row=0, column=0, sticky='ns')
        self._build_places()
        listing = ttk.Frame(browser)
        listing.grid(row=0, column=1, sticky='nsew')
        listing.columnconfigure(0, weight=1)
        listing.rowconfigure(0, weight=1)
        self.tree = ttk.Treeview(listing, columns=('kind', 'size', 'modified'),
                                 selectmode='extended' if self.mode == 'multiple' else 'browse')
        for key, label, width, anchor in (('#0', 'Nom', 310, 'w'), ('kind', 'Type', 80, 'w'),
                                          ('size', 'Taille', 100, 'e'), ('modified', 'Modifié le', 150, 'w')):
            column = 'name' if key == '#0' else key
            self.tree.heading(key, text=label, anchor=anchor, command=lambda c=column: self._sort(c))
            self.tree.column(key, width=width, minwidth=65, anchor=anchor, stretch=(key == '#0'))
        self.tree.grid(row=0, column=0, sticky='nsew')
        ys = ttk.Scrollbar(listing, orient='vertical', command=self.tree.yview)
        ys.grid(row=0, column=1, sticky='ns')
        xs = ttk.Scrollbar(listing, orient='horizontal', command=self.tree.xview)
        xs.grid(row=1, column=0, sticky='ew')
        self.tree.configure(yscrollcommand=ys.set, xscrollcommand=xs.set)
        self.tree.bind('<<TreeviewSelect>>', self._selection_changed)
        self.tree.bind('<Double-1>', self._activate)
        self.tree.bind('<Return>', self._activate)
        self.tree.bind('<BackSpace>', lambda event: self._up())
        self.tree.bind('<Control-a>', self._select_all)
        self.tree.bind('<space>', self._preview_key)
        details = ttk.Frame(self, padding=(10, 8))
        details.grid(row=3, column=0, sticky='ew')
        details.columnconfigure(0, weight=1)
        ttk.Label(details, textvariable=self.details_var, wraplength=660).grid(row=0, column=0, sticky='w')
        self.preview_button = ttk.Button(details, text='Écouter', command=self._preview, state='disabled')
        self.preview_button.grid(row=0, column=1, padx=(8, 0))
        bottom = ttk.Frame(self, padding=(10, 0, 10, 10))
        bottom.grid(row=4, column=0, sticky='ew')
        bottom.columnconfigure(1, weight=1)
        ttk.Label(bottom, text='Dossier :' if self.mode == 'directory' else 'Nom :').grid(row=0, column=0, sticky='w', padx=(0, 6))
        self.name_entry = ttk.Entry(bottom, textvariable=self.name_var)
        self.name_entry.grid(row=0, column=1, sticky='ew', pady=(0, 5))
        ttk.Label(bottom, text='Type :').grid(row=1, column=0, sticky='w', padx=(0, 6))
        self.filter_box = ttk.Combobox(bottom, textvariable=self.filter_var, values=self._filter_labels, state='readonly')
        self.filter_box.grid(row=1, column=1, sticky='ew')
        self.filter_box.bind('<<ComboboxSelected>>', self._filter_changed)
        if self.mode == 'directory':
            self.filter_box.configure(state='disabled')
        buttons = ttk.Frame(bottom)
        buttons.grid(row=0, column=2, rowspan=2, sticky='ns', padx=(10, 0))
        self.accept_button = ttk.Button(buttons, text={'save': 'Enregistrer', 'directory': 'Choisir'}.get(self.mode, 'Ouvrir'), command=self.accept)
        self.accept_button.pack(fill='x', pady=(0, 5))
        ttk.Button(buttons, text='Annuler', command=self.cancel).pack(fill='x')
        ttk.Label(bottom, textvariable=self.status_var, anchor='w').grid(row=2, column=0, columnspan=3, sticky='ew', pady=(8, 0))

    def _build_places(self):
        for child in self.places.winfo_children():
            child.destroy()
        ttk.Label(self.places, text='Emplacements').pack(anchor='w', pady=(0, 6))
        locations = [('Accueil', str(Path.home())), ('Racine', Path(self.current_dir).anchor or os.sep)]
        locations += list(_SHORTCUTS())
        favorites = self.preferences.get('favorites', [])
        if isinstance(favorites, list):
            locations += [(Path(p).name or p, p) for p in favorites if isinstance(p, str)]
        listing = ttk.Frame(self.places)
        listing.pack(fill='both', expand=True)
        tree = ttk.Treeview(listing, show='tree', selectmode='browse', height=8)
        tree.column('#0', width=145, minwidth=100, stretch=True)
        tree.pack(side='left', fill='both', expand=True)
        scrollbar = ttk.Scrollbar(listing, orient='vertical', command=tree.yview)
        scrollbar.pack(side='right', fill='y')
        tree.configure(yscrollcommand=scrollbar.set)
        destinations = {}
        seen = set()
        for label, path in locations:
            path = str(path)
            if path in seen:
                continue
            seen.add(path)
            iid = tree.insert('', 'end', text=str(label))
            destinations[iid] = path
        def select(event):
            selected = tree.selection()
            if selected:
                self.navigate(destinations[selected[0]])
        tree.bind('<<TreeviewSelect>>', select)
        ttk.Separator(self.places).pack(fill='x', pady=8)
        ttk.Button(self.places, text='Dossier favori', width=18, command=self._favorite).pack(fill='x')

    def _position(self):
        self.update_idletasks()
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        width, height = min(980, sw - 40), min(610, sh - 80)
        saved = self.preferences.get('size', [])
        if isinstance(saved, list) and len(saved) == 2 and all(isinstance(n, int) for n in saved):
            width, height = min(max(saved[0], 700), sw - 40), min(max(saved[1], 420), sh - 80)
        self.minsize(min(700, width), min(420, height))
        x = max(0, min(sw - width, self.owner.winfo_rootx() + (self.owner.winfo_width() - width) // 2))
        y = max(0, min(sh - height - 40, self.owner.winfo_rooty() + (self.owner.winfo_height() - height) // 2))
        self.geometry(f'{width}x{height}+{x}+{y}')

    def _preference_key(self):
        return self.mode + ':' + '|'.join(','.join(p) for label, p in self.filters)

    def navigate(self, path, remember=True, history_index=None):
        if self._closed:
            return
        path = os.path.expanduser(os.path.expandvars(str(path)))
        path = os.path.abspath(path if os.path.isabs(path) else os.path.join(self.current_dir, path))
        self._scan_stop.set()
        self._scan_stop = threading.Event()
        self._scan_token += 1
        self._pending_history = (remember, history_index)
        self._loading = True
        if self.mode == 'directory':
            self.name_var.set('')
        self.accept_button.configure(state='disabled')
        self.path_var.set(path)
        self.status_var.set('Chargement du dossier…')
        self._stop_preview()
        self._audio_token += 1
        threading.Thread(target=_scan_directory, args=(path, self._scan_token, self._scan_stop, self._results), daemon=True).start()

    def _drain(self):
        for unused in range(12):
            try:
                item = self._results.get_nowait()
            except queue.Empty:
                break
            if item[0] == 'scan' and item[1] == self._scan_token:
                _, token, path, entries, error = item
                self._loading = False
                self.accept_button.configure(state='normal')
                if error:
                    self.path_var.set(self.current_dir)
                    self.status_var.set('Dossier inaccessible.')
                    messagebox.showerror('Ouverture du dossier', error, parent=self)
                    continue
                self.current_dir, self._entries = path, entries
                self.path_var.set(path)
                remember, index = self._pending_history
                if index is not None:
                    self._history_index = index
                elif remember and (not self._history or self._history[self._history_index] != path):
                    self._history = self._history[:self._history_index + 1] + [path]
                    self._history_index = len(self._history) - 1
                self.back_button.configure(state='normal' if self._history_index > 0 else 'disabled')
                self.forward_button.configure(state='normal' if self._history_index < len(self._history) - 1 else 'disabled')
                self.search_var.set('')
                self._render()
            elif item[0] == 'audio' and item[1] == self._audio_token:
                info = item[3]
                if info:
                    minutes, seconds = divmod(round(info['duration'], 1), 60)
                    duration = f'{int(minutes)} min {seconds:04.1f} s'
                    channels = {1: 'mono', 2: 'stéréo'}.get(info['channels'], f"{info['channels']} canaux")
                    self.details_var.set(f"{Path(item[2]).name}\n{duration}  ·  {info['rate']:,} Hz  ·  {channels}".replace(',', ' '))
                else:
                    self.details_var.set(f'{Path(item[2]).name}\nInformations audio indisponibles.')
        code = self.player.poll()
        if code is not None:
            self.preview_button.configure(text='Écouter')
            if code:
                self.status_var.set('Lecture audio impossible pour ce fichier ou ce périphérique.')
        self._later(40, self._drain)

    def _render(self):
        if self._closed:
            return
        self._render_token += 1
        token = self._render_token
        selected = {self._rows[i].path for i in self.tree.selection() if i in self._rows}
        children = self.tree.get_children()
        if children:
            self.tree.delete(*children)
        self._rows = {}
        index = max(self.filter_box.current(), 0)
        patterns = self.filters[index][1]
        entries = _visible_entries(self._entries, patterns, self.search_var.get(), self.hidden_var.get(), self._sort_column, self._sort_reverse)
        if self.mode == 'directory':
            entries = [e for e in entries if e.is_dir]
        if not self._loading:
            self.status_var.set(f'{len(entries)} élément(s)')
        def insert(offset=0):
            if token != self._render_token:
                return
            for position, entry in enumerate(entries[offset:offset + 150], start=offset):
                iid = f'r{position}'
                try:
                    date = datetime.fromtimestamp(entry.modified).strftime('%d/%m/%Y %H:%M')
                except (ValueError, OSError, OverflowError):
                    date = ''
                self.tree.insert('', 'end', iid=iid, text=entry.name,
                                 values=(entry.kind, '' if entry.is_dir else _size_label(entry.size), date))
                self._rows[iid] = entry
                if entry.path in selected:
                    self.tree.selection_add(iid)
            if offset + 150 < len(entries):
                self._later(1, lambda: insert(offset + 150))
        insert()

    def _sort(self, column):
        self._sort_reverse = not self._sort_reverse if column == self._sort_column else False
        self._sort_column = column
        for key, label in (('#0', 'Nom'), ('kind', 'Type'), ('size', 'Taille'), ('modified', 'Modifié le')):
            active = ('name' if key == '#0' else key) == column
            self.tree.heading(key, text=label + ((' ▼' if self._sort_reverse else ' ▲') if active else ''))
        self._render()

    def _selection(self):
        return [self._rows[i] for i in self.tree.selection() if i in self._rows]

    def _selection_changed(self, event=None):
        self._audio_token += 1
        self._stop_preview()
        entries = self._selection()
        files = [e for e in entries if not e.is_dir]
        if files:
            # StringVar.get would stringify a Python tuple as its Python repr.
            # Tcl's format produces an actual Tcl list string, including all
            # quoting needed for spaces, braces, backslashes and newlines.
            names = tuple(e.name for e in files)
            # The trailing literal forces Tcl to create a string instead of
            # returning the original list object through tkinter's wantobjects.
            self.name_var.set(self.tk.call('format', '%s ', names)[:-1]
                              if self.mode == 'multiple' else files[0].name)
            self._selected_text = self.name_var.get()
        elif self.mode == 'directory' and entries:
            self.name_var.set(entries[0].name)
        elif entries and self.mode in ('open', 'multiple'):
            self.name_var.set('')
        self.preview_button.configure(state='disabled')
        if len(files) == 1 and Path(files[0].name).suffix.lower() in AUDIO_EXTENSIONS:
            path, token = files[0].path, self._audio_token
            self.details_var.set(f'{files[0].name}\nLecture des informations audio…')
            self.preview_button.configure(state='normal')
            self._later(180, lambda: self._start_probe(path, token))
        elif files:
            self.details_var.set(f'{len(files)} fichier(s) sélectionné(s)  ·  {_size_label(sum(e.size for e in files))}')
        elif entries:
            self.details_var.set(entries[0].path)
        else:
            self.details_var.set('Sélectionnez un fichier pour afficher ses informations.')

    def _start_probe(self, path, token):
        if token != self._audio_token:
            return
        if self._metadata_worker is not None and self._metadata_worker.is_alive():
            self._later(100, lambda: self._start_probe(path, token))
            return
        self._metadata_worker = threading.Thread(target=_probe_worker, args=(path, token, self._results), daemon=True)
        self._metadata_worker.start()

    def _stop_preview(self):
        self.player.stop()
        self.preview_button.configure(text='Écouter')

    def _preview(self):
        if self.player.process is not None:
            self._stop_preview()
            return
        files = [e for e in self._selection() if not e.is_dir]
        if len(files) != 1:
            return
        try:
            _BEFORE_PREVIEW()
            self.player.start(files[0].path)
            self.preview_button.configure(text='Arrêter')
        except OSError as exc:
            self.status_var.set(str(exc))

    def _preview_key(self, event):
        if str(self.preview_button['state']) != 'disabled':
            self._preview()
        return 'break'

    def _activate(self, event=None):
        if event is not None and getattr(event, 'num', None) == 1:
            iid = self.tree.identify_row(event.y)
            if not iid:
                return 'break'
            self.tree.selection_set(iid)
        entries = self._selection()
        if len(entries) == 1 and entries[0].is_dir:
            if self.mode == 'directory':
                self.name_var.set('')
            self.navigate(entries[0].path)
        else:
            self.accept()
        return 'break'

    def _select_all(self, event=None):
        if self.mode == 'multiple':
            self.tree.selection_set([i for i, e in self._rows.items() if not e.is_dir])
        return 'break'

    def _filter_changed(self, event=None):
        variable = self.options.get('typevariable')
        if variable is not None:
            variable.set(self.filters[max(0, self.filter_box.current())][0])
        self._render()

    def _focus_address(self, event=None):
        self.address.focus_set()
        self.address.selection_range(0, 'end')
        return 'break'

    def _address_enter(self, event=None):
        self.navigate(self.path_var.get())
        return 'break'

    def _toggle_hidden(self, event=None):
        self.hidden_var.set(not self.hidden_var.get())
        self._render()
        return 'break'

    def _up(self):
        self.name_var.set('' if self.mode == 'directory' else self.name_var.get())
        self.navigate(str(Path(self.current_dir).parent))
        return 'break'

    def _history_move(self, step):
        index = self._history_index + step
        if 0 <= index < len(self._history):
            if self.mode == 'directory':
                self.name_var.set('')
            self.navigate(self._history[index], remember=False, history_index=index)
        return 'break'

    def _favorite(self):
        favorites = self.preferences.get('favorites', [])
        favorites = [p for p in favorites if isinstance(p, str)] if isinstance(favorites, list) else []
        if self.current_dir in favorites:
            favorites.remove(self.current_dir)
        else:
            favorites.append(self.current_dir)
        self.preferences['favorites'] = favorites[-12:]
        self._build_places()

    def _new_folder(self):
        name = simpledialog.askstring('Nouveau dossier', 'Nom du dossier :', parent=self)
        if not name:
            return
        if name in ('.', '..') or '/' in name or '\\' in name:
            messagebox.showerror('Nouveau dossier', 'Saisissez un nom de dossier simple.', parent=self)
            return
        try:
            path = Path(self.current_dir) / name
            path.mkdir()
            self.navigate(str(path))
        except OSError as exc:
            messagebox.showerror('Nouveau dossier', str(exc), parent=self)

    def _resolve(self, name):
        name = os.path.expanduser(str(name))
        return os.path.abspath(name if os.path.isabs(name) else os.path.join(self.current_dir, name))

    def accept(self):
        if self._loading or self._closed:
            return 'break'
        text = self.name_var.get()
        entries = self._selection()
        if self.mode == 'directory':
            path = self._resolve(text) if text else self.current_dir
            if os.path.exists(path) and not os.path.isdir(path):
                self.status_var.set('Sélectionnez un dossier.')
                return 'break'
            if self.options.get('mustexist', True) and not os.path.isdir(path):
                self.status_var.set('Ce dossier n’existe pas.')
                return 'break'
            self._finish(path)
            return 'break'
        if not text:
            if len(entries) == 1 and entries[0].is_dir:
                self.navigate(entries[0].path)
            else:
                self.status_var.set('Sélectionnez un fichier ou saisissez son nom.')
            return 'break'
        path = self._resolve(text)
        if os.path.isdir(path):
            self.navigate(path)
            return 'break'
        if self.mode == 'save':
            path = _with_extension(path, self.options.get('defaultextension'))
            if not os.path.isdir(os.path.dirname(path)):
                self.status_var.set('Le dossier de destination n’existe pas.')
                return 'break'
            if os.path.isdir(path):
                self.status_var.set('Ce nom désigne un dossier.')
                return 'break'
            if os.path.exists(path) and self.options.get('confirmoverwrite', True):
                if not messagebox.askyesno('Remplacer le fichier ?', f'{Path(path).name}\n\nCe fichier existe déjà. Le remplacer ?', parent=self):
                    return 'break'
            self._finish(path)
            return 'break'
        if self.mode == 'multiple':
            selected = [e.path for e in entries if not e.is_dir]
            if selected and text == self._selected_text:
                paths = selected
            elif os.path.isfile(path):
                paths = [path]
            else:
                try:
                    paths = [self._resolve(p) for p in self.tk.splitlist(text)]
                except tk.TclError:
                    paths = []
        else:
            paths = [path]
        if not paths or any(not os.path.isfile(p) for p in paths):
            self.status_var.set('Un des fichiers demandés est introuvable.')
            return 'break'
        self._finish(tuple(paths) if self.mode == 'multiple' else paths[0])
        return 'break'

    def _finish(self, result):
        self.result = result
        self.cancel()

    def cancel(self):
        if self._closed:
            return 'break'
        self.preferences['hidden'] = self.hidden_var.get()
        self.preferences['size'] = [self.winfo_width(), self.winfo_height()]
        last_dirs = self.preferences.get('last_dirs', {})
        if not isinstance(last_dirs, dict):
            last_dirs = {}
        last_dirs[self._preference_key()] = self.current_dir
        self.preferences['last_dirs'] = last_dirs
        _save_preferences(self.preferences)
        self._cleanup()
        self.destroy()
        return 'break'

    def _cleanup(self):
        if self._closed:
            return
        self._closed = True
        self._scan_stop.set()
        self._audio_token += 1
        self._render_token += 1
        self.player.stop()
        for job in tuple(self._jobs):
            try:
                self.after_cancel(job)
            except tk.TclError:
                pass
        self._jobs.clear()

    def _destroyed(self, event):
        if event.widget is self:
            self._cleanup()


def _choose(mode, **options):
    global _ACTIVE
    empty = () if mode == 'multiple' else ''
    if _ACTIVE is not None:
        try:
            if _ACTIVE.winfo_exists():
                _ACTIVE.lift()
                _ACTIVE.focus_set()
                return empty
        except tk.TclError:
            pass
    root = _PARENT() or tk._default_root
    parent = options.pop('parent', None)
    temporary_root = None
    if root is None and parent is not None and not isinstance(parent, str):
        root = parent._root()
    if root is None:
        temporary_root = root = tk.Tk()
        root.withdraw()  # Only our own temporary root, never an application GUI.
    if isinstance(parent, str):
        parent = root.nametowidget(parent)
    owner = (parent or root.focus_get() or root).winfo_toplevel()
    previous_focus = owner.focus_get()
    previous_grab = owner.grab_current()
    previous_grab_status = previous_grab.grab_status() if previous_grab is not None else None
    dialog = None
    try:
        dialog = StudioFileDialog(owner, mode, **options)
        _ACTIVE = dialog
        dialog.wait_visibility()
        dialog.grab_set()
        dialog.lift()
        (dialog.name_entry if mode == 'save' else dialog.tree).focus_set()
        owner.wait_window(dialog)
        return dialog.result
    except tk.TclError:
        # A destroyed application or editor cancels its child selection.
        return empty
    finally:
        _ACTIVE = None
        if dialog is not None:
            try:
                dialog._cleanup()
                if dialog.winfo_exists():
                    dialog.destroy()
            except tk.TclError:
                pass
        try:
            if previous_grab is not None and previous_grab.winfo_exists():
                if previous_grab_status == 'global':
                    previous_grab.grab_set_global()
                else:
                    previous_grab.grab_set()
            if previous_focus is not None and previous_focus.winfo_exists():
                previous_focus.focus_set()
        except tk.TclError:
            pass
        if temporary_root is not None:
            temporary_root.destroy()


def askopenfilename(**options):
    return _choose('open', **options)


def askopenfilenames(**options):
    return _choose('multiple', **options)


def asksaveasfilename(**options):
    return _choose('save', **options)


def askdirectory(**options):
    return _choose('directory', **options)
