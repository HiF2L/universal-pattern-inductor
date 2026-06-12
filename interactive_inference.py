# pyrefly: ignore [missing-import]
import torch
import os
import sys
from model import IQMicroUnit
from logger import OP_NAMES
from execution_engine import ExecutionEngine

# ANSI colors for premium console look
C_BLUE = "\033[94m"
C_GREEN = "\033[92m"
C_YELLOW = "\033[93m"
C_RED = "\033[91m"
C_CYAN = "\033[96m"
C_BOLD = "\033[1m"
C_END = "\033[0m"

def print_header(title: str):
    print(f"\n{C_BOLD}{C_BLUE}{'=' * 60}")
    print(f" {title.center(58)}")
    print(f"{'=' * 60}{C_END}")

def parse_vector(prompt: str) -> torch.Tensor:
    while True:
        try:
            val = input(prompt).strip()
            if not val:
                return None
            # Support space or comma separation
            val = val.replace(",", " ")
            parts = [float(x) for x in val.split()]
            if len(parts) != 5:
                print(f"{C_RED}[-] Ошибка: Вектор должен состоять ровно из 5 чисел. Повторите ввод.{C_END}")
                continue
            return torch.tensor(parts, dtype=torch.float32)
        except ValueError:
            print(f"{C_RED}[-] Ошибка: Неверный формат чисел. Повторите ввод.{C_END}")

def print_vector(t: torch.Tensor) -> str:
    vals = []
    for val in t.tolist():
        if abs(val - round(val)) < 1e-4:
            vals.append(str(int(round(val))))
        else:
            vals.append(f"{val:.2f}")
    return "[" + ", ".join(vals) + "]"

def run_interactive():
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print_header("Интерактивный Помощник: Логический Вывод IQ-Модели (Phase 6.1)")
    print(f"{C_CYAN}[+] Загрузка модели на устройство: {device}...{C_END}")
    
    # Load model
    model_path = "cot_best_model.pt"
    if not os.path.exists(model_path):
        print(f"{C_RED}[-] Ошибка: Чекпоинт '{model_path}' не найден. Сначала запустите обучение.{C_END}")
        return
        
    model = IQMicroUnit(hidden_dim=256)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        print(f"{C_GREEN}[+] Модель успешно загружена!{C_END}")
    except Exception as e:
        print(f"{C_RED}[-] Не удалось загрузить модель: {e}{C_END}")
        return
        
    while True:
        print_header("Шаг 1: Сбор Контекстных Примеров")
        print(f"{C_CYAN}Пожалуйста, введите до 4-х примеров. Оставьте ввод пустым для завершения сбора.{C_END}")
        
        X_context_list = []
        Y_context_list = []
        
        for i in range(4):
            print(f"\n{C_BOLD}--- Пример {i+1} ---{C_END}")
            x = parse_vector("  Введите входной вектор X (5 чисел через пробел или запятую): ")
            if x is None:
                if len(X_context_list) < 1:
                    print(f"{C_YELLOW}[!] Предупреждение: Нужно ввести хотя бы один пример!{C_END}")
                    # Force repeat
                    x = parse_vector("  Введите входной вектор X (5 чисел): ")
                    if x is None:
                        continue
                else:
                    break
            
            y = parse_vector("  Введите соответствующий таргет Y (5 чисел): ")
            while y is None:
                print(f"{C_RED}[-] Ошибка: Таргет не может быть пустым.{C_END}")
                y = parse_vector("  Введите соответствующий таргет Y (5 чисел): ")
                
            X_context_list.append(x)
            Y_context_list.append(y)
            
        T_ctx = len(X_context_list)
        print(f"\n{C_GREEN}[+] Собрано примеров: {T_ctx}. Начинаем сессию запросов.{C_END}")
        
        # Prepare tensors for batch evaluation
        X_ctx_tensor = torch.stack(X_context_list, dim=0).unsqueeze(0).to(device) # [1, T_ctx, 5]
        Y_ctx_tensor = torch.stack(Y_context_list, dim=0).unsqueeze(0).to(device) # [1, T_ctx, 5]
        
        while True:
            print_header("Шаг 2: Ввод Запроса (Query)")
            print(f"{C_CYAN}Введите тестовый вектор X_query (или введите 'new' для новых примеров, 'exit' для выхода):{C_END}")
            raw_input = input("  Query X: ").strip().lower()
            if raw_input == 'exit':
                print(f"\n{C_YELLOW}[*] Выход из программы. До встречи!{C_END}")
                sys.exit(0)
            elif raw_input == 'new' or not raw_input:
                break
                
            try:
                # Parse manually
                raw_input = raw_input.replace(",", " ")
                parts = [float(x) for x in raw_input.split()]
                if len(parts) != 5:
                    print(f"{C_RED}[-] Ошибка: Вектор должен состоять ровно из 5 чисел.{C_END}")
                    continue
                X_qry_tensor = torch.tensor(parts, dtype=torch.float32).unsqueeze(0).to(device) # [1, 5]
            except ValueError:
                print(f"{C_RED}[-] Ошибка: Неверный формат чисел.{C_END}")
                continue
                
            # Perform Autoregressive Inference
            print(f"\n{C_BOLD}{C_YELLOW}--- Размышления и цепочка рассуждений (Chain-of-Thought) ---{C_END}")
            
            with torch.no_grad():
                current_program = torch.full((1, 1), 25, dtype=torch.long, device=device) # [1, 1] (START=25)
                pred_tokens_list = []
                
                for step in range(5):
                    logits = model(X_ctx_tensor, Y_ctx_tensor, X_qry_tensor, program_tokens=current_program) # [1, S, 26]
                    next_token_logits = logits[0, -1, :] # [26]
                    probs = torch.softmax(next_token_logits, dim=-1) # [26]
                    
                    # Top-3 Probabilities analysis
                    top_probs, top_indices = torch.topk(probs, k=3)
                    top_3_str = []
                    for val, i_tok in zip(top_probs.tolist(), top_indices.tolist()):
                        name = OP_NAMES.get(i_tok, f"OP_{i_tok}")
                        top_3_str.append(f"{C_BOLD}{name}{C_END}: {val*100:.2f}%")
                        
                    next_token = torch.argmax(next_token_logits, dim=-1).item()
                    selected_name = OP_NAMES.get(next_token, f"OP_{next_token}")
                    
                    print(f"  {C_CYAN}Шаг {step+1}:{C_END} предсказано {C_GREEN}{selected_name}{C_END} | Top-3 варианты: {', '.join(top_3_str)}")
                    
                    pred_tokens_list.append(next_token)
                    current_program = torch.cat([current_program, torch.tensor([[next_token]], device=device)], dim=1)
                    
            # Execute on virtual engine
            result_y = ExecutionEngine.execute_chain(X_qry_tensor[0], pred_tokens_list)
            
            print_header("Результат Выполнения")
            print(f"  {C_BOLD}Входной Query X:  {C_END}{print_vector(X_qry_tensor[0])}")
            print(f"  {C_BOLD}Синтезированная цепочка:{C_END}")
            chain_ops_names = [OP_NAMES.get(tok, f"OP_{tok}") for tok in pred_tokens_list if tok != 0]
            print(f"    {' -> '.join(chain_ops_names)}")
            print(f"  {C_BOLD}Вычисленный Y:    {C_END}{C_GREEN}{print_vector(result_y)}{C_END}")
            print(f"{C_BLUE}{'=' * 60}{C_END}")

if __name__ == "__main__":
    try:
        run_interactive()
    except KeyboardInterrupt:
        print(f"\n{C_YELLOW}[*] Выход из программы по KeyboardInterrupt. До встречи!{C_END}")
