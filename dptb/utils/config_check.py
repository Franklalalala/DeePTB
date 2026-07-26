# this file is to check the input configuration file to run specific commands.

from dptb.utils.tools import j_loader, j_must_have
from typing import Optional
from dptb.utils.argcheck import normalize


#TODO: 对于loss 和 data option 的检查还没有完整

def check_config_train(
        INPUT,
        init_model: Optional[str],
        restart: Optional[str],
        **kwargs):
    
    if all((init_model, restart)):
        raise RuntimeError("--init-model and --restart should not be set at the same time")
    
    jdata = j_loader(INPUT)
    jdata = normalize(jdata)

    if not (restart or init_model):
        j_must_have(jdata, "model_options")
        j_must_have(jdata, "train_options")

    assert j_must_have(jdata["data_options"], "train"), "train data set in data_options is not provided in the input configuration file."
    train_data_config = jdata["data_options"]["train"]

    if train_data_config.get("get_eigenvalues") and not train_data_config.get("get_Hamiltonian"):
        assert jdata['train_options']['loss_options']['train'].get("method") in ["eigvals"]

    # if train_data_config.get("get_Hamiltonian") and not train_data_config.get("get_eigenvalues"):
    #     assert jdata['train_options']['loss_options']['train'].get("method").startswith("hamil")

    # if train_data_config.get("get_Hamiltonian") and train_data_config.get("get_eigenvalues"):
    #     raise RuntimeError("The train data set should not have both get_Hamiltonian and get_eigenvalues set to True.")

    #if jdata["data_options"].get("validation"):
    
    
    if not (restart or init_model):
        model_options = jdata["model_options"]
        if not all(
            (model_options.get("embedding"), model_options.get("prediction"))
        ):
            raise ValueError(
                "0726-light requires model_options.embedding and "
                "model_options.prediction."
            )
        
        #if jdata["data_options"].get("reference"):
        #    log.info("reference set is provided. Then the loss options should have set the reference loss options.")
