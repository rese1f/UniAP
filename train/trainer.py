import pytorch_lightning as pl
import torch
import torchvision.transforms as T
from einops import rearrange, repeat, reduce
import os
import numpy as np

from model.model_factory import get_model

from dataset.dataloader_factory import get_train_loader, get_validation_loaders, get_eval_loader
from dataset.utils import to_device, mix_fivecrop, crop_arrays

from .optim import get_optimizer
from .loss import compute_loss, compute_metric, AutomaticWeightedLoss
from .visualize import visualize_batch, postprocess_depth, postprocess_semseg

from scipy import ndimage



class LightningTrainWrapper(pl.LightningModule):
    def __init__(self, config, verbose=True, load_pretrained=True):
        '''
        Pytorch lightning wrapper for Visual Token Matching.
        '''
        super().__init__()

        # load model.
        self.model = get_model(config, verbose=verbose, load_pretrained=load_pretrained)
        self.config = config
        self.verbose = verbose
        self.awl = AutomaticWeightedLoss(num=3)

        # tools for validation.
        self.crop = T.Compose([
            T.FiveCrop(config.img_size),
            T.Lambda(lambda crops: torch.stack([crop for crop in crops]))
        ])

        # self.support_data = self.load_support_data()

        if self.config.stage == 1:
            for attn in self.model.matching_module.matching:
                attn.attn_dropout.p = self.config.attn_dropout
        
        # save hyper=parameters
        self.save_hyperparameters()

    def load_state_dict(self, state_dict, strict=True):
        # Handle loading of optimizer parameters separately
        if "awl.params" in state_dict:
          
            awl_params_tensor = state_dict["awl.params"]
            self.awl.params = torch.nn.Parameter(awl_params_tensor)

        # Load the rest of the state_dict using the Module's method
        super().load_state_dict(state_dict, strict=strict)


    def configure_optimizers(self):
        '''
        Prepare optimizer and lr scheduler.
        '''
        optimizer, self.lr_scheduler = get_optimizer(self.config, 
                                                     self.model, 
                                                     learnable_params = [{'params': self.awl.parameters(), 'lr': self.config.lr}])
        return optimizer
    
    def train_dataloader(self, verbose=True):
        '''
        Prepare training loader.
        '''
        return get_train_loader(self.config, verbose=(self.verbose and verbose))
    
    def val_dataloader(self, verbose=True):
        '''
        Prepare validation loaders.
        '''
      
        if not self.config.no_eval:
            # use external data from validation split
            if self.config.stage == 0:
             
                val_loaders, loader_tag = get_validation_loaders(self.config, verbose=(self.verbose and verbose))
                self.valid_tasks = list(val_loaders.keys())
                self.valid_tag = loader_tag
                
                return list(val_loaders.values())
            
            # use second half of support data as validation query
            else:
                val_loaders, loader_tag = get_validation_loaders(self.config, verbose=(self.verbose and verbose))
                self.valid_tasks = list(val_loaders.keys())
                self.valid_tag = loader_tag
                self.valid_tag = 'mtest_support'
                return list(val_loaders.values())
                
    def test_dataloader(self, verbose=True):
        '''
        Prepare test loaders.
        '''
        test_loader = get_eval_loader(self.config, self.config.task, split=self.config.test_split,
                                      channel_idx=self.config.channel_idx, verbose=(self.verbose and verbose))
        
        return test_loader
        
    def forward(self, *args, **kwargs):
        '''
        Forward data to model.
        '''
        return self.model(*args, **kwargs)
    
    def training_step(self, batch, batch_idx):
        '''
        A single training iteration.
        '''
        # forward model and compute loss.

        loss, v_loss = compute_loss(self.model, batch, self.config)

        if self.config.task == 'keypoints2d':
            total_loss = loss["loss_PE"]
        elif self.config.task == 'mask':
            total_loss = loss["loss_SS"]
        elif self.config.task == 'cls':
            total_loss = loss["loss_CLS"]
        else:
            total_loss = self.awl(loss["loss_PE"], loss["loss_SS"], loss["loss_CLS"])

        # schedule learning rate.
        self.lr_scheduler.step(self.global_step)

        if self.config.stage == 0:
            tag = ''

        elif self.config.stage == 1:
            tag = f'_{self.config.task}'

        # log losses and learning rate.
        log_dict = {
            f'training/loss{tag}': total_loss.detach(),
            f'training/lr{tag}': self.lr_scheduler.lr,
            f'training/w_pe': self.awl.params[0],
            f'training/w_ss': self.awl.params[1],
            f'training/w_cls': self.awl.params[2],
            'step': self.global_step,
        }
        for key, value in v_loss.items():
            if value != 0:
                log_dict.update([(f'training/{key}', value.detach())])
        

        
        self.log_dict(
            log_dict,
            logger=True,
            on_step=True,
            sync_dist=False,
        )

        return total_loss
    
    @torch.autocast(device_type='cuda', dtype=torch.float32)
    def inference(self, batch, task):
        # support data

        X, Y, M, t_idx, X_S, Y_S, M_S, _ = batch

        t_idx = t_idx.long()
    
        X = repeat(X, 'B C H W -> B T 1 C H W', T= t_idx.size(1) )
    
        # predict labels on each crop
        X_S = rearrange(X_S, '(B T) N C H W -> B T N C H W', T = t_idx.size(1))
        Y_S_in = torch.where(M_S.bool(), Y_S, torch.ones_like(Y_S) * self.config.mask_value)
        Y_S_in = rearrange(Y_S_in, '(B T)  N C H W -> B T N C H W', T = t_idx.size(1))

        Y_pred= self.model(X_S, Y_S_in, X, t_idx=t_idx, sigmoid=False)

        Y_pred = rearrange(Y_pred, 'B T N C H W -> (B T N) C H W')
       
        return Y_pred

    def calculate_pck(self, gt_heatmap, pred_heatmap, threshold = 0.2):
        """
        Calculate the Percentage of Correct Keypoints (PCK) metric.

        Args:
            gt_heatmap (numpy.ndarray): Ground truth heatmap with Gaussian distribution.
            pred_heatmap (numpy.ndarray): Predicted heatmap with Gaussian distribution.
            threshold (float): Threshold distance to determine correct detections.

        Returns:
            float: PCK value for the given heatmaps and threshold.
        """
        # Convert heatmaps to keypoints (coordinates of the maximum value)
        gt_keypoint = np.unravel_index(np.argmax(gt_heatmap), gt_heatmap.shape)
        pred_keypoint = np.unravel_index(np.argmax(pred_heatmap), pred_heatmap.shape)

        # Normalize keypoints to the range [0, 1]
        gt_keypoint_norm = (gt_keypoint[1] / float(gt_heatmap.shape[1]), gt_keypoint[0] / float(gt_heatmap.shape[0]))
        pred_keypoint_norm = (pred_keypoint[1] / float(pred_heatmap.shape[1]), pred_keypoint[0] / float(pred_heatmap.shape[0]))

        # Calculate Euclidean distance between keypoints
        distance = np.linalg.norm(np.array(gt_keypoint_norm) - np.array(pred_keypoint_norm))

        # Check if the distance is within the threshold
        correct_detection = distance <= threshold

        return correct_detection

    def calculate_result_pck(self, pck_res):
        # Create a new dictionary to store accuracy information for each class_name and joint_id
        accuracy_dict_class = {}
        accuracy_dict_joint = {}
        # Calculate accuracy for each class_name and joint_id
        for class_name_i, joint_id_i in pck_res.items():
            for joint_id, correct_detection_list in joint_id_i.items():
                accuracy = sum(correct_detection_list) / len(correct_detection_list)
                # Save accuracy information to the new dictionary
                accuracy_dict_class.setdefault(class_name_i, {})[joint_id] = accuracy

        for class_name_i, joint_id_i in pck_res.items():
            for joint_id, correct_detection_list in joint_id_i.items():
                accuracy = sum(correct_detection_list) / len(correct_detection_list)
                # Save accuracy information to the new dictionary
                accuracy_dict_joint.setdefault(joint_id, {})[class_name_i] = accuracy


        # Calculate the average accuracy across all classes and joints
        total_accuracy = sum([accuracy for class_accuracy in accuracy_dict_class.values() for accuracy in class_accuracy.values()]) / len([accuracy for class_accuracy in accuracy_dict_class.values() for accuracy in class_accuracy.values()])

        # Calculate the average accuracy for each class_name and joint_id
        avg_acc_class = {class_name_i: sum(class_accuracy.values()) / len(class_accuracy) for class_name_i, class_accuracy in accuracy_dict_class.items()}
        avg_acc_joint = {joint_id: sum(class_accuracy.values()) / len(class_accuracy) for joint_id, class_accuracy in accuracy_dict_joint.items()}
        # avg_acc_joint = {joint_id: sum(accuracy_dict[class_name_i].get(joint_id, 0.0) for class_name_i in accuracy_dict) / len(accuracy_dict) for joint_id in range(len(next(iter(accuracy_dict.values()))))}

        print("task: pck , Total Accuracy:", total_accuracy)
    
    def calculate_iou(self, box1, box2):
        intersection = np.logical_and(box1, box2)
        union = np.logical_or(box1, box2)
        iou = np.sum(intersection) / np.sum(union)
        return iou

    def calculate_mAP(self, heatmap_predictions, ground_truth_heatmaps, threshold=0.5, iou_threshold=0.5):
        num_samples = len(heatmap_predictions)
        average_precision = 0

        for i in range(num_samples):
            prediction_heatmap = heatmap_predictions[i, 0]
            gt_heatmap = ground_truth_heatmaps[i, 0]

            # Apply threshold to create binary detections
            binary_detection = prediction_heatmap > threshold

            # Perform connected component analysis
            detected_objects, num_detected = ndimage.label(binary_detection)

            true_positives = 0
            false_positives = 0

            for detected_label in range(1, num_detected + 1):
                detected_object = detected_objects == detected_label

                max_iou = 0

                for gt_label in range(1, num_detected + 1):
                    gt_object = gt_heatmap == gt_label

                    iou = self.calculate_iou(detected_object, gt_object)

                    if iou > max_iou:
                        max_iou = iou

                if max_iou >= iou_threshold:
                    true_positives += 1
                else:
                    false_positives += 1

            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / np.sum(gt_heatmap)

            average_precision += precision

        average_precision /= num_samples
        return average_precision

    def on_validation_start(self) -> None:
        self.pck_res_02 = {}
        self.pck_res_005 = {}
        return super().on_validation_start()
    
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        '''
        Evaluate few-shot performance on validation dataset.
        '''
     
        # return 0, 1
        task = self.valid_tasks[dataloader_idx]
        # get query data
        X, Y, M, _, _, _, _, meta_info= batch

        # few-shot inference based on support data
        Y_pred = self.inference(batch, task)

        Y_pred = Y_pred.sigmoid()
        # discretization for semantic segmentation
        if 'mask' in task:
            Y_pred = (Y_pred > self.config.semseg_threshold).float()
        
        if 'keypoints2d' in task:
            # calculate pck
            class_name = meta_info[0]
            joint_id = meta_info[1]
            for i, (class_name_i, joint_id_i) in enumerate(zip(class_name, joint_id)):
                correct_detection_pck_02 = self.calculate_pck(Y[i].squeeze().float().cpu().numpy(), Y_pred[i].squeeze().float().cpu().numpy(), threshold=0.2)
                correct_detection_pck_005 = self.calculate_pck(Y[i].squeeze().float().cpu().numpy(), Y_pred[i].squeeze().float().cpu().numpy(), threshold=0.05)
                if class_name_i not in self.pck_res_02:
                    self.pck_res_02[class_name_i] = {}
                if joint_id_i not in self.pck_res_02[class_name_i]:
                    self.pck_res_02[class_name_i][joint_id_i] = []
                self.pck_res_02[class_name_i][joint_id_i].append(correct_detection_pck_02)

                if class_name_i not in self.pck_res_005:
                    self.pck_res_005[class_name_i] = {}
                if joint_id_i not in self.pck_res_005[class_name_i]:
                    self.pck_res_005[class_name_i][joint_id_i] = []
                self.pck_res_005[class_name_i][joint_id_i].append(correct_detection_pck_005)
                # self.calculate_result_pck(self.pck_res_02)
                # self.calculate_result_pck(self.pck_res_005)
        
        # compute evaluation metric
        metric = compute_metric(Y, Y_pred, M, task)
        metric *= len(X)
    
        # visualize first batch
        if batch_idx == 0 and "cls" not in task:
            X_vis = rearrange(self.all_gather(X), 'G B ... -> (B G) ...')
            Y_vis = rearrange(self.all_gather(Y), 'G B ... -> (B G) ...')
            M_vis = rearrange(self.all_gather(M), 'G B ... -> (B G) ...')
            Y_pred_vis = rearrange(self.all_gather(Y_pred), 'G B ... -> (B G) ...')
            vis_batch = (X_vis, Y_vis, M_vis, Y_pred_vis)
            self.vis_images(vis_batch, task)
        
        return metric, torch.tensor(len(X), device=self.device)
        
    def validation_epoch_end(self, validation_step_outputs):
        '''
        Aggregate losses of all validation datasets and log them into tensorboard.
        '''
        if len(self.valid_tasks) == 1:
            validation_step_outputs = (validation_step_outputs,)
        avg_loss = []
        log_dict = {'step': self.global_step}

        for task, losses_batch in zip(self.valid_tasks, validation_step_outputs):
            N_total = sum([losses[1] for losses in losses_batch])
            loss_pred = sum([losses[0] for losses in losses_batch])
            N_total = self.all_gather(N_total).sum()
            loss_pred = self.all_gather(loss_pred).sum()

            loss_pred = loss_pred / N_total

            log_dict[f'{self.valid_tag}/{task}_pred'] = loss_pred
            avg_loss.append(loss_pred)
        
            if 'keypoints2d' in task: 
                self.calculate_result_pck(self.pck_res_02)
                self.calculate_result_pck(self.pck_res_005)
            if 'mask' in task:
                print("MASK miou: ", loss_pred.item())
            if 'cls' in task:
                print("cls acc: ", loss_pred.item())
                       
        # log task-averaged error
        if self.config.stage == 0:
            avg_loss = sum(avg_loss) / len(avg_loss)
            log_dict[f'summary/{self.valid_tag}_pred'] = avg_loss
        self.log_dict(
            log_dict,
            logger=True,
            rank_zero_only=True
        )
       
        
    def on_test_start(self) -> None:
        self.pck_res_02 = {}
        self.pck_res_005 = {}
        return super().on_test_start()
    
    def test_step(self, batch, batch_idx):
        '''
        Evaluate few-shot performance on test dataset.
        '''
        task = self.config.task
        
        # get query data
        X, Y, M, _, _, _, _, meta_info= batch

        # few-shot inference based on support data
        Y_pred = self.inference(batch, task)

        Y_pred = Y_pred.sigmoid()
        # discretization for semantic segmentation
        if 'mask' in task:
            Y_pred = (Y_pred > self.config.semseg_threshold).float()
        
        if 'keypoints2d' in task:
            # calculate pck
            class_name = meta_info[0]
            joint_id = meta_info[1]
            for i, (class_name_i, joint_id_i) in enumerate(zip(class_name, joint_id)):
                correct_detection_pck_02 = self.calculate_pck(Y[i].squeeze().float().cpu().numpy(), Y_pred[i].squeeze().float().cpu().numpy(), threshold=0.2)
                correct_detection_pck_005 = self.calculate_pck(Y[i].squeeze().float().cpu().numpy(), Y_pred[i].squeeze().float().cpu().numpy(), threshold=0.05)
                if class_name_i not in self.pck_res_02:
                    self.pck_res_02[class_name_i] = {}
                if joint_id_i not in self.pck_res_02[class_name_i]:
                    self.pck_res_02[class_name_i][joint_id_i] = []
                self.pck_res_02[class_name_i][joint_id_i].append(correct_detection_pck_02)

                if class_name_i not in self.pck_res_005:
                    self.pck_res_005[class_name_i] = {}
                if joint_id_i not in self.pck_res_005[class_name_i]:
                    self.pck_res_005[class_name_i][joint_id_i] = []
                self.pck_res_005[class_name_i][joint_id_i].append(correct_detection_pck_005)

            # self.calculate_result_pck(self.pck_res_02)
            # self.calculate_result_pck(self.pck_res_005)
        
        # compute evaluation metric
        metric = compute_metric(Y, Y_pred, M, task, self.miou_evaluator, self.config.stage)
        metric *= len(X)
    
        
        return metric, torch.tensor(len(X), device=self.device)
    
    def test_epoch_end(self, test_step_outputs):
        # append test split to save_postfix
        log_name = f'result{self.config.save_postfix}_split:{self.config.test_split}.pth'
        log_path = os.path.join(self.config.result_dir, log_name)
        
       
        if self.config.task == 'keypoints2d' : 
            print("EPOCH END:")
            self.calculate_result_pck(self.pck_res_02)
            self.calculate_result_pck(self.pck_res_005)

        N_total = sum([losses[1] for losses in test_step_outputs])
        metric = sum([losses[0] for losses in test_step_outputs]) / N_total
        metric = metric.cpu().item()
        print("metric: ",metric)
        torch.save(metric, log_path)
        
    @pl.utilities.rank_zero_only
    

    def vis_images(self, batch, task, vis_shot=-1, **kwargs):
        '''
        Visualize query prediction into tensorboard.
        '''
        X, Y, M, Y_pred = batch

        # choose proper subset.
        if vis_shot > 0:
            X = X[:vis_shot]
            Y = Y[:vis_shot]
            M = M[:vis_shot]
            Y_pred = Y_pred[:vis_shot]
        
        # set task-specific post-processing function for visualization
        postprocess_fn = None
        

        # visualize batch
       
        vis = visualize_batch(X, Y, M, Y_pred, task, postprocess_fn=postprocess_fn, **kwargs)
        self.logger.experiment.add_image(f'{self.valid_tag}/{task}', vis, self.global_step)