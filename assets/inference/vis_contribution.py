
# ! wget https://huggingface.co/cis-lmu/glotlid/resolve/main/model.bin

import fasttext
import numpy as np
import re
import string
from copy import deepcopy
import plotly.graph_objects as go

class LID:
    """A class for language identification."""
    
    def __init__(self, model_path, languages=-1):
        """Initialize the MaskLID class.
        
        Args:
            model_path (str): The path to the fastText model.
            languages (int or list, optional): The indices or list of language labels to consider. Defaults to -1.
        """
        self.model = fasttext.load_model(model_path)
        self.output_matrix = self.model.get_output_matrix()
        self.labels = self.model.get_labels()
        self.language_indices = self._compute_language_indices(languages)
        self.labels = [self.labels[i] for i in self.language_indices]

    def _compute_language_indices(self, languages):
        """Compute indices of selected languages.
        
        Args:
            languages (int or list): The indices or list of language labels.
            
        Returns:
            list: Indices of selected languages.
        """
        if languages != -1 and isinstance(languages, list):
            return [self.labels.index(l) for l in set(languages) if l in self.labels]
        return list(range(len(self.labels)))

    def _softmax(self, x):
        """Compute softmax values for each score in array x.
        
        Args:
            x (numpy.ndarray): Input array.
            
        Returns:
            numpy.ndarray: Softmax output.
        """
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

    def _normalize_text(self, text):
        """Normalize input text.
        
        Args:
            text (str): Input text.
            
        Returns:
            str: Normalized text.
        """
        replace_by = " "
        # replacement_map = {ord(c): replace_by for c in '\n_:' + '•#{|}' + string.digits}
        replacement_map = {ord(c): replace_by for c in '\n'}
        text = text.translate(replacement_map)
        return re.sub(r'\s+', ' ', text).strip()

    def _get_color(self, value, min_val, max_val):
        """ Scale color dynamically based on value range with emphasis on distances """

        # Normalize value to the range [0, 1] based on the min and max logits
        scaled_value = (value - min_val) / (max_val - min_val)  # Normalize to [0, 1]
        scaled_value = max(0, min(1, scaled_value))  # Ensure it's within bounds [0, 1]

        if value >= 0:
            # For positive values, intensify green as value increases towards max
            green_intensity = int(255 - (scaled_value * 255))  # Decrease green intensity as value increases
            return f'rgba(0, {green_intensity}, 0, 1)'  # Green intensifies as value increases
        else:
            # For negative values, intensify red as value decreases towards min
            red_intensity = int(255 - (scaled_value * 255))  # Increase red intensity as value decreases
            return f'rgba({red_intensity}, 0, 0, 1)'  # Red intensifies as value decreases

    def predict(self, text, k=1):
        """Predict the language of the input text.
        
        Args:
            text (str): Input text.
            k (int, optional): Number of top predictions to retrieve. Defaults to 1.
            
        Returns:
            tuple: Top predicted labels and their probabilities.
        """
        sentence_vector = self.model.get_sentence_vector(text)
        result_vector = np.dot(self.output_matrix, sentence_vector)
        softmax_result = self._softmax(result_vector)[self.language_indices]
        top_k_indices = np.argsort(softmax_result)[-k:][::-1]
        top_k_labels = [self.labels[i] for i in top_k_indices]
        top_k_probs = softmax_result[top_k_indices]
        return tuple(top_k_labels), top_k_probs

    def compute_v(self, sentence_vector):
        """Compute the language vectors for a given sentence vector.
        
        Args:
            sentence_vector (numpy.ndarray): Sentence vector.
            
        Returns:
            list: Sorted list of labels and their associated vectors.
        """
        result_vector = np.dot(self.output_matrix[self.language_indices, :], sentence_vector)
        return sorted(zip(self.labels, result_vector), key=lambda x: x[1], reverse=True)

    def compute_v_per_word(self, text):
        """Compute language vectors for each word in the input text.
        
        Args:
            text (str): Input text.
            
        Returns:
            dict: Dictionary containing language vectors for each word.
        """
        words = self.model.get_line(text)[0]
        # words = [w for w in words if w not in ['</s>', '</s>']]
        subword_ids = [self.model.get_subwords(sw)[1] for sw in words]
        sentence_vector = [np.sum([self.model.get_input_vector(id) for id in sid], axis=0) for sid in subword_ids]

        dict_text = {}
        for i, word in enumerate(words):
            key = f"{i}_{word}"
            dict_text[key] = {'logits': self.compute_v(sentence_vector[i])}

        return dict_text


    def predict_ner(self, text, k=2, limit_labels = None):
                
        # all labels
        self.labels = self.model.get_labels()
        
        ## limit labels
        if limit_labels:
            self.labels = list(limit_labels)
        else:
            limit_labels, _ = self.predict(text, k=k)
            limit_labels = list(limit_labels)
            self.labels = limit_labels
        
        v_data = self.compute_v_per_word(text)

        ## all labels
        self.labels = self.model.get_labels()
    
        return limit_labels, v_data
        
    def vis_predict_ner(self, text, k=2, limit_labels=None):
        """Visualize NER predictions with colors and logits displayed on hover."""

        data_labels, data = self.predict_ner(text, k=k, limit_labels=limit_labels)

        words = []
        logits = []
        hovertext = []

        # Collect all logits across the text to determine min and max
        for idx, (word, details) in enumerate(data.items()):
            words.append(word.split('_')[1])  # Extract the word from '0_Hi', '1_this', etc.
            hover_info = []
            for label in data_labels:
                score = next((logit[1] for logit in details['logits'] if logit[0] == label), 0)
                logits.append(score)
                hover_info.append(f"{label}: {score:.2f}")
            hovertext.append("<br>".join(hover_info))  # Display logits for each label on hover

        # Find the global min and max logits for scaling
        min_val = min(logits)
        max_val = max(logits)

        # Prepare color data for each label and each word
        colors_dict = {label: [] for label in data_labels}

        for idx, (word, details) in enumerate(data.items()):
            for label in data_labels:
                score = next((logit[1] for logit in details['logits'] if logit[0] == label), 0)
                colors_dict[label].append(self._get_color(score, min_val, max_val))

        # Create a plotly scatter plot to visualize the words with their respective colors
        fig = go.Figure()

        # Add a trace for each label with a different y-position to avoid overlap
        for i, label in enumerate(data_labels):
            fig.add_trace(go.Scatter(
                x=list(range(len(words))),
                y=[i] * len(words),  # Each label gets its own y-position
                mode='text',
                text=words,
                textfont=dict(size=14, color=colors_dict[label]),
                marker=dict(color=colors_dict[label], size=20),
                name=label,
                showlegend=True,
                hovertext=hovertext,  # Display hovertext when mouse hovers
            ))

        # Update the layout for better appearance and zoom
        fig.update_layout(
            title="Text Visualization for All Labels",
            xaxis=dict(
                tickvals=list(range(len(words))),
                ticktext=words,
                tickangle=45,
                range=[-1, len(words)],  # Ensure the plot is scaled according to the number of words
                showgrid=False,
                zeroline=False,
                autorange=False,  # Allow zoom functionality on the x-axis
            ),
            yaxis=dict(
                showgrid=False,
                zeroline=False,
                tickvals=list(range(len(data_labels))),
                ticktext=data_labels,
            ),
            plot_bgcolor="white",
            showlegend=True,
            height=400,
            autosize=True,  # Allow the plot to adjust size automatically
            dragmode="pan",  # Enable pan by default, while allowing zoom
        )

        fig.show()



m = LID('model.bin')


text = "أجي بعدة واش تقد تعطيني لورديناتور غدا"
text = """قولو ل مرات يوسف بلي غدي نديرو عرس ولد ختي مويا ف نهار سبعة حنا لي جينا نطلو على خالي
"""
text = """هاد لغنى كيف تنسميوه ف الوسط الفني حمار السوق
قال لي ملي تعرضها علي يتعجب بها ولكن شي واحد عادي غيعتبرها عادية
"""
text = "Hi ali is coming home, but who are you?"

text = """هاد لغنى كيف تنسميوه ف الوسط الفني حمار السوق"""

text = """يعتقد الأطباء أن مدة غياب لاعب وسط منتخب إنجلترا ونادي بورنموث «لويس كوك» ستصل إلى تسعة أشهر بسبب تلك لإصابة التي آلمت به في الركبة خلال فوز فريقه على هدرسفيلد تاون في الجولة الـ 15 من البريميرليج منتصف الأسبوع الماضي.
# وقال النادي بعد يومين من المباراة التي أقيمت مساء الثلاثاء "فحصًا بالأشعة كشف إصابة كوك بقطع سيء في الرباط الصليبي للركبة اليمنى".
# وعبر المدير الفني للنادي "إيدي هاو" عن حزنه، قائلاً "الجميع هنا يشعر بحزن عميق لما حدث للويس. إنها ضربة قوية للاعب موهوب للغاية كان يشكل جزءًا مهمًا من فريقنا خلال آخر 18 شهرًا".
# لويس كوك /21 عامًا/ هو قائد منتخب شباب إنجلترا، وانضم إلى بورنموث قادمًا من ليدز يونايتد في يوليو 2016، وقد شارك في 15 مباراة مع ناديه الحالي هذا الموسم.
# وتقدم بورنموث إلى المركز السابع في جدول ترتيب الدوري خلف إيفرتون.
# اقرأ ايضًا"""

text = """
الاتحاد البيضاوي يهزم وداد فاس بثلاثية
فاز الاتحاد البيضاوي , على ضيفه وداد فاس بثلاثة أهداف مقابل هدف واحد، في المباراة التي جمعت الفريقين اليوم السبت، على ارضية ملعب الزوالي في الدار البيضاء، برسم الجولة التاسعة من البطولة الوطنية الاحترافية الدرجة الثانية.
ومكن هذا الفوز الطاس من الارتقاء إلى المركز السابع بـ 11 نقطة، في حين تجمد رصيد الفريق الفاسي عند النقطة 9 في المرتبة 13.
"""


text = """
17 من جمادى الآخرة 1362هـ = 20 من يونيو 1943م: مولد الشيخ محمد صفوت نور الدين الرئيس العام لجماعة أنصار السنة المحمدية في مصر. ولد بمدينة بلبيس التابعة لمحافظة الشرقية، عمل بالتربية والتعليم حتى صار مديرًا عامًّا، وتولى رئاسة جماعة أنصار السنة المحمدية بعد وفاة الشيخ محمد علي عبد الرحيم؛ ليكون سادس رؤساء الجماعة.
"""

text = """
Watch Wotaku ni Koi wa Muzukashii Watch Wotaku ni Koi wa Muzukashii English Sub Watch Wotaku ni Koi wa Muzukashii English Subtitle Wotaku ni Koi wa Muzukashii Wotaku ni Koi wa Muzukashii English Subbed Wotaku ni Koi wa Muzukashii English Subtitle Wotaku ni Koi wa Muzukashii HD Wotaku ni Koi wa Muzukashii Free Download Anime On going Anime List W Comedy Anime Romance Anime
Fast Download ( Full HD )
"""

text = """
الليبراليه (انجليزى: Liberalism; فرنساوى: Libéralisme) اشتقت الكلمه من ليبر liber هيا كلمه لاتينى معناها " الحر ". الليبراليه هى مذهب فكرى و سياسى واقتصادى. الليبراليه بتطبق حسب أخلاق و ظروف المجتمع والبلد اللى بتطبقها بس مبنيه على اسس واحده من اهمها ان الواحد حر يعمل اللى هو عايزه طالما مش بيئذى او بيضر حد تانى ، و مش من حق الدوله و لا اى جماعة دينيه و لا اى جهه تانيه تتدخل فى حياته الشخصيه تحت أى حجه. الليبراليه السياسيه بمعناها العام ان كل واحد فى المجتمع حر ومن حقه ينضم لأى حزب او جماعه طالما شرعيه و مش بتهدد المجتمع و بتطالب بتعدد الاحزاب و ان كل واحد يقول رأيه السياسى بحريه طالما معداش حدود الادب والاخلاق اللى فى المجتمع اللى عايش فيه ، و بتطالب كمان بالمدنيه وتأيد النظام الديمقراطى و الانتخبات للحصول على السلطة و تداولها. الليبرالية كمان بتنادى ان كل واحد حر يتبع الدين او المذهب اللى عايزه و ان دى بتعتبر حرية شخصيه للانسان ، ومش المفروض الدين يستخدم فى السياسه و عشان اغراض سياسيه. الليبرالية بتحترم حرية وكرامة و حقوق الانسان، وبتعتبر ان حقوق الانسان مالهاش اى حدود الا حقوق الانسان التاني.
"""


text = m._normalize_text(text)


m.predict(text, k=5)[0]


m.vis_predict_ner(text, k=2)


d = m.predict_ner(text, limit_labels=['ary_Arab', 'arb_Arab'])  
m.vis_predict_ner(text, limit_labels= ['ary_Arab', 'arb_Arab'])





