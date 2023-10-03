from dataset_manager import DataDB, Dataset
import requests
import wandb
import tqdm
import os


class Trainner:
    def __init__(
            self, map_db_dir="./dataset_db", server="http://0.0.0.0:3000/"
    ) -> None:
        self.db = DataDB(map_db_dir)
        self.server = server
        wandb.init(project="ssg_training_protocol")
        self.fault_time = 0

    def train(self, epoch, n_epoch_save_ckpt=1):
        for e in range(epoch):
            try:
                batch = self.db.get_batch()
                running_loss = 0
                loop = tqdm.tqdm(enumerate(batch, 0))
                for i, step_data in loop:
                    flow = step_data["data"]
                    # for entity in flow:
                    package = {
                        "messages": [
                            {
                                "role": list(entity.keys())[0],
                                "text": list(entity.values())[0],
                            }
                            for entity in flow
                        ]
                    }
                    loss = requests.post(
                        self.server + "train/tx-data", json=package
                    ).json()["loss"]
                    running_loss += loss

                    loop.set_description(
                        f"epoch:[{e + 1}/{epoch}],step:[{i + 1}/{len(batch)}]"
                    )
                    loop.set_postfix(loss=loss, loss_smooth=running_loss / (i + 1))

                    wandb.log({"loss": loss, "loss_smooth": running_loss / (i + 1)})
                if e % n_epoch_save_ckpt == 0:
                    requests.post(self.server + "/train/save-weight",
                                  json={"model_name": f"ssg_training_protocol_epoch_{e}"})

            except:
                self.fault_time += 1
                if self.fault_time == 20:
                    raise Exception("The maximum number of failures was reached.")
                print(f"[Warning] Training Process Faulted, Total: {self.fault_time}")


if __name__ == "__main__":
    dir = os.path.dirname(os.path.realpath(__file__))
    parent = os.path.dirname(dir)
    ckpt_dir = os.path.join(parent, "checkpoints")
    if not os.path.exists(ckpt_dir):
        os.mkdir(ckpt_dir)
    t = Trainner("./dataset_db")
    t.train(100, n_epoch_save_ckpt=5)
