from prompt_toolkit import prompt
import os
import requests
from ssg.ssg_utils import (
    Conversation,
    apply_train,
    inferance,
    regenerate,
    conversation_hist,
    save,
    clear_state,
)


user = "User"
bot = "AI"
if __name__ == "__main__":
    resp_c = ""
    while True:
        msg = prompt("\n>")
        if msg.lower() == "/apply":
            loss = apply_train()
            save("online_training")
            print(f"loss={loss}")
        elif msg.lower() == "/full_apply":
            while loss > 0.5:
                loss = apply_train()
                print(f"loss={loss}")
            save("online_training-full")
        elif msg == "+":
            regenerate()
        elif msg == "/clrstate":
            clear_state()
        elif msg.lower() == "/hist":
            for c in conversation_hist:
                print(c())
        elif msg.lower() == "/savmodel":
            save("online_training")
        elif msg[:3].lower() == "/g ":
            real_msg = msg[3:]
            cs = [Conversation(role='text', text=real_msg)]
            resp_c = inferance(
                cs,
                free_gen=True,
            )
        elif msg[:6].lower() == "/book ":
            real_msg = msg[6:]
            cs = [Conversation(role='common', text=real_msg)]
            resp_c = inferance(
                cs,
                to_char=bot,
                target_role="summary",
            )
        else:
            entity = msg.split("--")
            msg = entity[0]
            free_gen = False
            role = "conversation"
            to_char = bot
            fr = user
            ni = False
            if len(entity) > 1:
                for e in entity[1:]:
                    if e.strip().lower() == "free":
                        free_gen = True
                    if e.strip().lower() == "ni":
                        ni = True
                    if e[:5].lower == "from ":
                        fr = e[5:].strip()
                    if e[:3].lower == "to ":
                        to_char = e[3:].strip()
                    if e[:5].lower == "role ":
                        role = e[5:].strip()

            cs = (
                [Conversation(role=role, text=f"{fr}: {msg}")]
                if not ni
                else [Conversation(role=role, text=msg)]
            )
            resp_c = inferance(
                cs,
                free_gen=free_gen,
                to_char=to_char,
                target_role=role,
            )
