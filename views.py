# Pls remember that you should use transformers==4.33
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.views import View
import json
import pandas as pd
import torch
from transformers import NllbTokenizer, AutoModelForSeq2SeqLM, get_constant_schedule_with_warmup
from transformers.optimization import Adafactor
from sacremoses import MosesPunctNormalizer
import re
import unicodedata
from tqdm import tqdm

def preproc(text, mpn):
    clean = mpn.normalize(text)
    clean = replace_non_printing_char(clean)
    clean = unicodedata.normalize("NFKC", clean)
    return clean

def replace_non_printing_char(text):
    return ''.join(c if unicodedata.category(c) not in {"C", "Cc", "Cf", "Cs", "Co", "Cn"} else ' ' for c in text)

def fix_tokenizer(tokenizer, new_lang):
    old_len = len(tokenizer) - int(new_lang in tokenizer.added_tokens_encoder)
    tokenizer.lang_code_to_id[new_lang] = old_len-1
    tokenizer.id_to_lang_code[old_len-1] = new_lang
    tokenizer.fairseq_tokens_to_ids["<mask>"] = len(tokenizer.sp_model) + len(tokenizer.lang_code_to_id) + tokenizer.fairseq_offset
    tokenizer.fairseq_tokens_to_ids.update(tokenizer.lang_code_to_id)
    tokenizer.fairseq_ids_to_tokens = {v: k for k, v in tokenizer.fairseq_tokens_to_ids.items()}
    if new_lang not in tokenizer._additional_special_tokens:
        tokenizer._additional_special_tokens.append(new_lang)
    tokenizer.added_tokens_encoder = {}
    tokenizer.added_tokens_decoder = {}

@method_decorator(csrf_exempt, name='dispatch')
class TrainModelView(View):
    def post(self, request):
        try:
            data = json.loads(request.body)
            source_language = data.get('source_language')
            target_language = data.get('target_language')
            training_steps = int(data.get('training_steps', 32000))
            batch_size = int(data.get('batch_size', 16))
            max_length = int(data.get('max_length', 128))
            warmup_steps = int(data.get('warmup_steps', 1000))
            moses_lang = data.get('moses_lang', 'es')
            model_name = data.get('model_name', 'latest_model')
            csv_file = request.FILES.get('csv_file')
            if not csv_file:
                return JsonResponse({'error': 'CSV file is required'}, status=400)
            df = pd.read_csv(csv_file)
            self.train_nllb_model(
                df,
                source_language,
                target_language,
                training_steps,
                batch_size,
                max_length,
                warmup_steps,
                moses_lang,
                model_name
            )

            return JsonResponse({'message': 'Model training completed'}, status=200)
        
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    def train_nllb_model(self, df, source_lang, target_lang, training_steps, batch_size, max_length, warmup_steps, moses_lang, model_name):
        tokenizer = NllbTokenizer.from_pretrained('facebook/nllb-200-distilled-600M')
        fix_tokenizer(tokenizer, f"{source_lang}_Latn")
        model = AutoModelForSeq2SeqLM.from_pretrained('facebook/nllb-200-distilled-600M').cuda()

        mpn = MosesPunctNormalizer(lang=moses_lang)
        mpn.substitutions = [(re.compile(r), sub) for r, sub in mpn.substitutions]

        df['source'] = df[source_lang].apply(lambda x: preproc(x, mpn))
        df['target'] = df[target_lang].apply(lambda x: preproc(x, mpn))

        optimizer = Adafactor(
            [p for p in model.parameters() if p.requires_grad],
            scale_parameter=False,
            relative_step=False,
            lr=1e-4,
            clip_threshold=1.0,
            weight_decay=1e-3,
        )
        scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)

        model.train()
        for step in tqdm(range(training_steps)):
            batch = df.sample(batch_size)
            
            tokenizer.src_lang = f"{source_lang}_Latn"
            x = tokenizer(batch['source'].tolist(), return_tensors='pt', padding=True, truncation=True, max_length=max_length).to(model.device)
            
            tokenizer.src_lang = f"{target_lang}_Latn"
            y = tokenizer(batch['target'].tolist(), return_tensors='pt', padding=True, truncation=True, max_length=max_length).to(model.device)
            y.input_ids[y.input_ids == tokenizer.pad_token_id] = -100

            loss = model(**x, labels=y.input_ids).loss
            loss.backward()

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            if step % 1000 == 0:
                print(f"Step {step}, Loss: {loss.item()}")

        model.save_pretrained(model_name)
        tokenizer.save_pretrained(model_name)
        print(f"Model saved as {model_name}")