'''
app.py

To launch:
python app.py
'''

import tkinter as tk
from tkhtmlview import HTMLScrolledText
import torch
from models import MODEL_DIR, load_model
from data.dataset import ReviewsDataset, EssaysDataset
from inference import Inference_IG, sample_example
import os

DATASET_WIDGET_OPTIONS = ['reviews', 'essays']
DEFAULT_MODEL_DICT = {'reviews': 'basic_transformer_reviews', 'essays': 'basic_transformer_essays'}
TEXT_LENGTH_WIDGET_DICT = {'Short (< 1,000)': 1000, 'Long (> 1,000)': [1000, 6000], 'Custom': None}
MAX_TEXT_LENGTH = 10000
DEFAULT_EXAMPLE_IDS = {'reviews': 0, 'essays': 0}
COLOR_MAPS = {'reviews': {1: 'red', 2: 'orange', 3: 'yellow', 4: 'lawngreen', 5: 'green'},
              'essays': {1: 'red', 2: 'orange', 3: 'yellow', 4: 'green'}}

class App:
    def __init__(self):
        # Runtime default options
        default_dataset_selection = DATASET_WIDGET_OPTIONS[0]
        default_model_selection = DEFAULT_MODEL_DICT[default_dataset_selection]
        default_text_length_selection = list(TEXT_LENGTH_WIDGET_DICT.keys())[0]

        # Global variables
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.model = None
        self.dataset = None
        self.category = None
        self.text_length = TEXT_LENGTH_WIDGET_DICT[default_text_length_selection]
        self.example_id_selection = DEFAULT_EXAMPLE_IDS[default_dataset_selection]
        self.example_id_display = None
        self.model_manually_selected = False
        self._update_model_selection(default_model_selection, dataset=default_dataset_selection)
        self._update_dataset_selection(default_dataset_selection)
        self.max_example_id = len(self.dataset)

        # Root and Canvas
        N_COLUMNS, N_ROWS = 3, 5
        self.root = tk.Tk()
        self.root.title("Inference App")
        for i in range(N_COLUMNS):
            self.root.columnconfigure(i, weight=1)
        for i in range(N_ROWS):
            self.root.rowconfigure(i, weight=1)
        self.canvas = tk.Canvas(self.root, width=650, height=650, bg='gray')
        self.canvas.grid(columnspan=N_COLUMNS, rowspan=N_ROWS, sticky=tk.NSEW)

        # Text HTML frame
        self.html_display_frame = tk.Frame(self.root)
        self.html_display_frame.grid(columnspan=3, row=1, sticky=tk.EW)
        self.html_display_frame.rowconfigure(0, weight=1)
        self.html_display_frame.columnconfigure(0, weight=1)
        self.html_display = HTMLScrolledText(self.html_display_frame)
        self.html_display.grid(row=0, sticky=tk.NSEW)
        self.current_example_label_strval = tk.StringVar(self.html_display_frame, value="Current Example id:  " + str(self.example_id_display))
        self.current_example_label = tk.Label(self.html_display_frame, textvariable=self.current_example_label_strval)
        self.current_example_label.grid(row=2, sticky="w")

        # Predictions frame
        self.predictions_frame = tk.Frame(self.root)
        self.predictions_frame.grid(columnspan=4, row=0, sticky="s")
        self.prediction_label = tk.Label(self.predictions_frame, text="Prediction: ", font=("TkDefaultFont", 20))
        self.prediction_label.grid(column=0, row=0)
        self.prediction_box = BoxWidget(self.predictions_frame)
        self.prediction_box.grid(column=1, row=0)
        self.actual_label = tk.Label(self.predictions_frame, text="Actual: ", font=("TkDefaultFont", 20))
        self.actual_label.grid(column=3, row=0)
        self.actual_box = BoxWidget(self.predictions_frame)
        self.actual_box.grid(column=4, row=0)

        # ----------------------------------------------------------------------------- #
        self.bottom_panel = tk.Frame(self.root)
        self.bottom_panel.grid(columnspan=4, rowspan=8, row=2, sticky="n")

        # Select model widget
        self.select_model_widget_frame = tk.Frame(self.bottom_panel)
        self.select_model_widget_frame.grid(rowspan=2, column=0, row=6)
        select_model_widget_label = tk.Label(self.select_model_widget_frame, text="Model")
        select_model_widget_label.grid(row=0, sticky='w')
        self.select_model_widget_selection = tk.StringVar(self.root)
        options = self._get_select_model_options(dataset=default_dataset_selection)
        if len(options) > 0:
            selection = default_model_selection if default_model_selection in options else options[0]
            self.select_model_widget_selection.set(selection)
        self.select_model_widget = tk.OptionMenu(self.select_model_widget_frame, self.select_model_widget_selection, *options, command=self._on_select_model_widget_action)
        self.select_model_widget.config(bg='#e0edea', height=1, width=48)
        self.select_model_widget.grid(row=1)

        # Select dataset widget
        self.select_dataset_widget_frame = tk.Frame(self.bottom_panel)
        self.select_dataset_widget_frame.grid(rowspan=2, column=1, row=6)
        select_dataset_widget_label = tk.Label(self.select_dataset_widget_frame, text="Dataset")
        select_dataset_widget_label.grid(row=0, sticky='w')
        self.select_dataset_widget_selection = tk.StringVar(self.root, value=default_dataset_selection)
        self.select_dataset_widget = tk.OptionMenu(self.select_dataset_widget_frame, self.select_dataset_widget_selection, *DATASET_WIDGET_OPTIONS, command=self._on_select_dataset_widget_action)
        self.select_dataset_widget.config(bg='#e0edea', height=1, width=8)
        self.select_dataset_widget.grid(row=1)

        # Go button
        self.go_button = tk.Button(self.bottom_panel, text="Go", command=self._on_go_button_action, bg='#e0edea', fg='black', height=3, width=10)
        self.go_button.grid(column=0, row=0, sticky="s")

        # ------------------------- Sampling Options frame ------------------------- #
        self.sampling_options_frame = tk.Frame(self.bottom_panel)
        self.sampling_options_frame.grid(rowspan=8, column=2, row=0)
        sampling_options_frame_label = tk.Label(self.sampling_options_frame, text="Sampling Options")
        sampling_options_frame_label.grid(row=0)

        # Select example ID widget
        self.select_example_id_frame = tk.Frame(self.sampling_options_frame)
        self.select_example_id_frame.grid(columnspan=2, rowspan=2, row=1)
        self.select_example_id_label = tk.Label(self.select_example_id_frame, text="Example id:")
        self.select_example_id_label.grid(columnspan=2, row=0, sticky="w")
        self.select_example_id_widget_selection = tk.IntVar(self.root)
        self.select_example_id_widget_selection.trace("w", self._on_select_example_id_widget_action)
        self.select_example_id_widget = tk.Entry(self.select_example_id_frame, textvariable=self.select_example_id_widget_selection, width=10)
        self.select_example_id_widget.configure( validate="key", validatecommand=(self.root.register(self._intvar_entry_validator(lower=0, upper=lambda: self.max_example_id)), "%P") )
        self.select_example_id_widget.grid(column=0, row=1)
        self.select_example_id_random_button = tk.Button(self.select_example_id_frame, text="Random", command=self._select_example_id_random_button_action, bg='#e0edea', fg='black')
        self.select_example_id_random_button.grid(column=1, row=1)

        # Select text length widget
        self.select_text_length_widget_frame = tk.Frame(self.sampling_options_frame)
        self.select_text_length_widget_frame.grid(rowspan=3, row=3)
        self.select_text_length_widget_frame_label = tk.Label(self.select_text_length_widget_frame, text="Text Length")
        self.select_text_length_widget_frame_label.grid(row=0, sticky='w')
        self.select_text_length_widget_selection = tk.StringVar(self.root, value=default_text_length_selection)
        self.select_text_length_widget = tk.OptionMenu(self.select_text_length_widget_frame, self.select_text_length_widget_selection, *TEXT_LENGTH_WIDGET_DICT.keys(), command=self._on_select_text_length_widget_action)
        self.select_text_length_widget.config(bg='#e0edea', height=1, width=16)
        self.select_text_length_widget.grid(row=1)
        self.select_text_length_custom_entry_frame = tk.Frame(self.select_text_length_widget_frame)
        self.select_text_length_custom_entry_frame.grid(columnspan=4, rowspan=1)
        self.select_text_length_custom_label_LB = tk.Label(self.select_text_length_custom_entry_frame, text="Low:")
        self.select_text_length_custom_label_RB = tk.Label(self.select_text_length_custom_entry_frame, text="High:")
        self.select_text_length_custom_label_LB.grid(column=0, row=1)
        self.select_text_length_custom_label_RB.grid(column=2, row=1)
        self.select_text_length_custom_entry_selection_LB = tk.IntVar(self.root)
        self.select_text_length_custom_entry_selection_RB = tk.IntVar(self.root)
        self.select_text_length_custom_entry_selection_RB.set(MAX_TEXT_LENGTH)
        self.select_text_length_custom_entry_selection_LB.trace("w", self._on_select_text_length_custom_entry_action)
        self.select_text_length_custom_entry_selection_RB.trace("w", self._on_select_text_length_custom_entry_action)
        self.select_text_length_custom_entry_LB = tk.Entry(self.select_text_length_custom_entry_frame, textvariable=self.select_text_length_custom_entry_selection_LB, width=5)
        self.select_text_length_custom_entry_RB = tk.Entry(self.select_text_length_custom_entry_frame, textvariable=self.select_text_length_custom_entry_selection_RB, width=5)
        self.select_text_length_custom_entry_LB.configure(validate="key", validatecommand=(self.root.register(self._intvar_entry_validator(lower=0, upper=MAX_TEXT_LENGTH)), "%P") )
        self.select_text_length_custom_entry_RB.configure(validate="key", validatecommand=(self.root.register(self._intvar_entry_validator(lower=0, upper=MAX_TEXT_LENGTH)), "%P") )
        self.select_text_length_custom_entry_LB.grid(column=1, row=1)
        self.select_text_length_custom_entry_RB.grid(column=3, row=1)
        self.select_text_length_custom_entry_frame.grid_remove()

        # Select category widget
        self.select_category_frame = tk.Frame(self.sampling_options_frame)
        self.select_category_frame.grid(rowspan=2, row=6)
        self.select_category_label = tk.Label(self.select_category_frame, text="Category:")
        self.select_category_label.grid(row=0, sticky="w")
        self.select_category_selection = tk.StringVar(self.root, value="All")
        self.select_category_widget = tk.OptionMenu(self.select_category_frame, self.select_category_selection, *self._get_unique_categories(), command=self._on_select_category_widget_action)
        self.select_category_widget.grid(row=1)

        self._update_inference_panel()


    def run(self):
        self.root.mainloop()

    def _update_model_selection(self, model_file=None, dataset=None):
        if not model_file:  model_file = self.select_model_widget_selection.get()
        if not dataset:  dataset = self.select_dataset_widget_selection.get()
        model_dict = load_model(model_name=os.path.join(dataset, model_file))
        self.model = model_dict['model']
        self.model.eval()

        if self.dataset:  self.dataset.load_tokenizer(model_dict['tokenizer'])

    def _update_dataset_selection(self, dataset):
        self.dataset = ReviewsDataset() if dataset == 'reviews' else EssaysDataset()
        self.max_example_id = len(self.dataset)

    def _update_category_selection(self, category):
        self.category = category

    def _update_selected_example_id(self, ex_id):
        self.example_id_selection = ex_id

    def _get_select_model_options(self, dataset=None):
        if not dataset:  dataset = self.select_dataset_widget_selection.get()
        return [f.split('.')[0] for f in os.listdir(os.path.join(MODEL_DIR, dataset)) if f.split('.')[1] == 'pt']

    def _get_unique_categories(self):
        return ['All'] + self.dataset.get_unique_categories().tolist()

    def _refresh_select_model_widget(self):
        def update_selection(value):
            self.select_model_widget_selection.set(value)
            self._on_select_model_widget_action(value)
        self.select_model_widget["menu"].delete(0, "end")
        for option in self._get_select_model_options():
            self.select_model_widget["menu"].add_command(label=option, command=lambda value=option: update_selection(value))

    def _refresh_select_category_widget(self):
        def update_selection(value):
            self.select_category_selection.set(value)
            self._on_select_category_widget_action(value)
        self.select_category_widget["menu"].delete(0, "end")
        for option in self._get_unique_categories():
            self.select_category_widget["menu"].add_command(label=option, command=lambda value=option: update_selection(value))

    def _update_inference_panel(self):
        self.example_id_display = self.example_id_selection
        self.current_example_label_strval.set("Current Example id:  " + str(self.example_id_display))
        IG_results = Inference_IG(self.model, self.dataset, self.example_id_display)
        self.html_display.set_html( IG_results['html'] )
        self.prediction_box.update(COLOR_MAPS[self.select_dataset_widget_selection.get()], IG_results['pred'])
        self.actual_box.update(COLOR_MAPS[self.select_dataset_widget_selection.get()], IG_results['actual'])

    def _intvar_entry_validator(self, lower=None, upper=None):
        def validator_func(value):
            if value == "":  return True
            if (type(value) == str  and  len(value) > 1  and  value[0] == "0"):
                return False
            try:
                value = int(value)
            except ValueError:
                return False
            if (lower != None) and value < (lower() if callable(lower) else lower):
                return False
            if (upper != None) and value > (upper() if callable(upper) else upper):
                return False
            return True
        return validator_func

    def _on_select_model_widget_action(self, selection):
        self._update_model_selection(selection)

    def _on_select_dataset_widget_action(self, selection):
        self._update_dataset_selection(selection)

        self.select_example_id_widget_selection.set(DEFAULT_EXAMPLE_IDS[selection])
        self._update_selected_example_id(DEFAULT_EXAMPLE_IDS[selection])

        self._refresh_select_model_widget()
        self.select_model_widget_selection.set(DEFAULT_MODEL_DICT[selection])
        self._update_model_selection(DEFAULT_MODEL_DICT[selection], dataset=selection)

        self._refresh_select_category_widget()
        self.select_category_selection.set('All')
        self._update_category_selection('All')

    def _on_select_text_length_widget_action(self, selection):
        if selection == "Custom":
            self.select_text_length_custom_entry_frame.grid(row=2)
        else:
            self.select_text_length_custom_entry_frame.grid_remove()
            self.text_length = TEXT_LENGTH_WIDGET_DICT[selection]

    def _on_select_text_length_custom_entry_action(self, *args):
        try:
            lower_bound = self.select_text_length_custom_entry_selection_LB.get()
        except:
            lower_bound = 0
        try:
            upper_bound = self.select_text_length_custom_entry_selection_RB.get()
        except:
            upper_bound = 0
        self.text_length = [lower_bound, upper_bound]

    def _on_select_example_id_widget_action(self, *args):
        try:
            example_id = self.select_example_id_widget_selection.get()
            self._update_selected_example_id(example_id)
        except:
            pass

    def _select_example_id_random_button_action(self):
        if (type(self.text_length) == list)  and  (self.text_length[0] > self.text_length[1]):
            raise Exception(f"Cannot Sample: custom range low ({self.text_length[0]}) is greater than high ({self.text_length[1]})")
        else:
            example_id = sample_example(self.dataset, filter_params={'length': self.text_length, 'category': self.category})
            self.select_example_id_widget_selection.set(example_id)
            self._update_selected_example_id(example_id)

    def _on_go_button_action(self):
        self._update_inference_panel()

    def _on_select_category_widget_action(self, selection):
        self._update_category_selection(selection)


class BoxWidget(tk.Canvas):
    def __init__(self, frame, width=40, height=40):
        super().__init__(frame, width=width, height=height)
        self.THICKNESS = 3
        self.textvar = tk.StringVar(self)
        start_coord = (self.THICKNESS / 2)
        self.create_rectangle(start_coord, start_coord, width, height, width=self.THICKNESS)
        self.create_text((start_coord + width) / 2, (start_coord + height) / 2, font=("TkDefaultFont", 15))

    def update(self, cmap, pred):
        fill_color = cmap[pred]
        self.textvar.set(pred)
        self.itemconfig(1, fill=fill_color)
        self.itemconfig(2, text=str(pred))


if __name__  == "__main__":
    App().run()