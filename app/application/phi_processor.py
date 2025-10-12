import re
import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from app.core.config import settings
from app.core.logging_config import logger
import time

class PhiProcessor:
    def __init__(self):
        logger.info("=" * 60)
        logger.info("Iniciando carga de modelo Phi-3-mini...")
        
        use_gpu = self._check_gpu_compatibility()
        
        if use_gpu:
            device = "cuda"
            dtype = torch.float16
            logger.info("Usando GPU para inferencia")
        else:
            device = "cpu"
            dtype = torch.float32
            logger.warning("Usando CPU para inferencia (más lento)")
        
        logger.info(f"Cargando tokenizer desde {settings.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            settings.model_name,
            trust_remote_code=True
        )
        
        logger.info(f"Cargando modelo en {device} con dtype={dtype}")
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                settings.model_name,
                torch_dtype=dtype,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            self.model = self.model.to(device)
        except Exception as e:
            logger.error(f"Error al cargar modelo en GPU: {e}")
            logger.info("Intentando cargar en CPU...")
            self.model = AutoModelForCausalLM.from_pretrained(
                settings.model_name,
                torch_dtype=torch.float32,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            self.model = self.model.to("cpu")
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.eval()
        
        device_info = next(self.model.parameters()).device
        
        if torch.cuda.is_available():
            logger.info(f"Memoria GPU usada: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
            logger.info(f"Memoria GPU reservada: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
        
        logger.info("=" * 60)

    def _check_gpu_compatibility(self):
        if not torch.cuda.is_available():
            logger.warning("CUDA no está disponible")
            return False
        
        try:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            compute_cap = torch.cuda.get_device_capability(0)
            
            logger.info(f"GPU detectada: {gpu_name}")
            logger.info(f"Memoria total: {gpu_memory:.2f} GB")
            logger.info(f"Compute Capability: {compute_cap[0]}.{compute_cap[1]}")
            
            test_tensor = torch.randn(100, 100).cuda()
            result = test_tensor @ test_tensor.T
            _ = result.cpu()
            del test_tensor, result
            torch.cuda.empty_cache()
            
            return True
        except Exception as e:
            logger.error(f"Error al verificar GPU: {e}")
            logger.warning("Fallando a CPU por seguridad")
            return False

    def tiene_errores_gramaticales(self, frase: str) -> list:
        """
        Detecta si la frase tiene posibles errores gramaticales que el modelo debe corregir.
        """
        errores = []
        
        # 1. Detectar "mucha/mucho" seguido de algo que no sea sustantivo plural
        if re.search(r'\bmucha?\s+(de\s+|individuo|persona|torpe|difícil|fácil)', frase, re.IGNORECASE):
            errores.append("concordancia mucho/mucha")
        
        # 2. Detectar redundancia sustantivo + adjetivo después de verbo ser/estar
        if re.search(r'\b(es|está|son|están|era|estaba)\s+(muy|mucha?|mucho)\s+(individuo|persona)\s+\w+', frase, re.IGNORECASE):
            errores.append("redundancia sustantivo+adjetivo")
        
        # 3. Detectar duplicaciones
        if re.search(r'\bde\s+de\b', frase, re.IGNORECASE):
            errores.append("duplicación 'de de'")
        if re.search(r'\b(\w+)ss\b', frase):
            errores.append("letra duplicada (ss)")
        
        # 4. Conjugaciones incorrectas obvias
        if re.search(r'\b\w+erar\b', frase):
            errores.append("conjugación incorrecta (-erar)")
        
        # 5. Artículos con género incorrecto
        if re.search(r'\bel\s+[a-záéíóúñ]+a\b', frase, re.IGNORECASE):
            errores.append("artículo masculino + sustantivo femenino")
        if re.search(r'\bla\s+[a-záéíóúñ]+o\b', frase, re.IGNORECASE):
            errores.append("artículo femenino + sustantivo masculino")
        
        return errores

    def corregir_gramatica(self, frase: str) -> str:
        start = time.perf_counter()
        
        try:
            # Detectar errores primero
            errores = self.tiene_errores_gramaticales(frase)
            
            if errores:
                logger.info(f"Errores detectados (regex): {', '.join(errores)}")
            else:
                logger.info("No se detectaron errores con regex, pero igual se corrige con el modelo.")
            
            messages = [
                {
                    "role": "system",
                    "content": (
                        "Eres un corrector gramatical de español. "
                        "Tu trabajo es detectar y corregir SOLO los errores gramaticales evidentes "
                        "sin cambiar el estilo ni el sentido.\n\n"
                        "INSTRUCCIONES CRÍTICAS:\n"
                        "1. Mantén TODAS las palabras de la frase salvo que sean redundantes ('individuo torpe'→'torpe').\n"
                        "2. NO cambies nombres propios, tiempos verbales, sujetos, pronombres, ni orden de palabras.\n"
                        "3. NO inventes ni elimines información. SOLO corrige lo mínimo.\n\n"
                        "ERRORES QUE DEBES CORREGIR:\n"
                        "- Concordancia de género/número: 'esa objeto' → 'ese objeto'.\n"
                        "- Mucho/mucha → muy cuando acompaña adjetivos: 'mucha individuo torpe' → 'muy torpe'.\n"
                        "- Eliminar sustantivos genéricos redundantes: 'individuo torpe' → 'torpe'.\n"
                        "- Duplicaciones: 'de de suerte' → 'de suerte', 'diligenciass' → 'diligencias'.\n"
                        "- Conjugaciones incorrectas por error de sufijo: 'perderar' → 'perder', 'hacerar' → 'hacer'.\n"
                        "- Artículos incorrectos por género: 'la carro' → 'el carro'.\n\n"
                        "NO CORRIJAS:\n"
                        "- Pronombres válidos (usted, tú, él, ella).\n"
                        "- Nombres propios (Santiago, María).\n"
                        "- Días o tiempos verbales correctos.\n\n"
                        "EJEMPLOS CRÍTICOS:\n"
                        "Entrada: 'esa objeto'\n"
                        "Errores: 'esa' debe concordar con 'objeto'.\n"
                        "Salida: 'ese objeto'\n\n"
                        "Entrada: 'mucha individuo torpe'\n"
                        "Errores: 'mucha'→'muy'; eliminar 'individuo'.\n"
                        "Salida: 'muy torpe'\n\n"
                        "Entrada: 'ese santiago es mucha individuo torpe, como se va a comer ese gol.'\n"
                        "Errores: 'mucha individuo torpe' → 'muy torpe'.\n"
                        "Salida: 'ese santiago es muy torpe, como se va a comer ese gol.'\n\n"
                        "Entrada: 'paseme esa objeto y arreglamos de una vez que mi joven de 5 años...'\n"
                        "Errores: 'esa objeto' → 'ese objeto'.\n"
                        "Salida: 'paseme ese objeto y arreglamos de una vez que mi joven de 5 años...'\n\n"
                        "Entrada: 'mamá manda decir que no te vayas a perderar perder en la fiesta de hoy.'\n"
                        "Errores: 'perderar' → 'perder'.\n"
                        "Salida: 'mamá manda decir que no te vayas a perder en la fiesta de hoy.'\n\n"
                        "Responde SOLO con la frase corregida final, nada más."
                    )
                },
                {
                    "role": "user",
                    "content": f"Corrige la gramática de la siguiente frase sin cambiar su sentido:\n{frase}\n\nCorrección:"
                }
            ]
            
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.1,
                    do_sample=False,
                    top_p=0.8,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                    repetition_penalty=1.05
                )
            
            generated = self.tokenizer.decode(
                outputs[0][len(inputs["input_ids"][0]):],
                skip_special_tokens=True
            ).strip()
            
            # Limpiar
            if '\n' in generated:
                generated = generated.split('\n')[0].strip()
            
            prefijos = ["Frase corregida:", "Salida:", "Respuesta:", "Resultado:", "Corrección:", "Output:"]
            for prefijo in prefijos:
                if generated.lower().startswith(prefijo.lower()):
                    generated = generated[len(prefijo):].strip()
            
            if generated.startswith('"') and generated.endswith('"'):
                generated = generated[1:-1]
            if generated.startswith("'") and generated.endswith("'"):
                generated = generated[1:-1]
            
            # Post-procesamiento: red de seguridad para eliminar redundancias
            generated = re.sub(r'\b(muy|mucho|mucha)\s+(individuo|persona)\s+(\w+)\b', r'muy \3', generated, flags=re.IGNORECASE)
            
            # Validaciones
            ratio = len(generated.split()) / len(frase.split())
            if ratio < 0.75 or ratio > 1.4:
                logger.warning(f"Cambio muy drástico (ratio: {ratio:.2f}), manteniendo versión con reemplazos")
                return frase
            
            palabras_orig = set(frase.lower().split())
            palabras_result = set(generated.lower().split())
            palabras_cambio_permitido = {'mucha', 'mucho', 'muy', 'de', 'el', 'la', 'un', 'una', 'individuo', 'persona'}
            palabras_clave = palabras_orig - palabras_cambio_permitido
            palabras_perdidas = palabras_clave - palabras_result
            
            if len(palabras_perdidas) > len(palabras_clave) * 0.3:
                logger.warning(f"Se perdieron palabras importantes: {palabras_perdidas}")
                return frase
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return generated.strip()
            
        finally:
            end = time.perf_counter()
            logger.info(f"Tiempo de procesamiento: {end - start:.2f} segundos")

    def __del__(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()