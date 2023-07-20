import torch


def print_log(inputs: str):
    print(f'LOG >>> {inputs}')
    

def get_device(GPU_NUM: str) -> torch.device:
    if torch.cuda.device_count() == 1:
        output = torch.device('cuda')
    elif torch.cuda.device_count() > 1:
        output = torch.device(f'cuda:{GPU_NUM}')
    else:
        output = torch.device('cpu')

    print_log(f'{output} is checked')
    return output

def get_log_name(args):
    
    log_list = {
        'time': args.current_time,
        'task': args.task,
        'batch': args.batch_size,
        'lr': args.lr,
        'window': args.window_length,
        'etc': args.log_etc,
    }
    
    output = log_list['time']
    
    for key in log_list.keys():
        if key == 'time' or key == 'etc': continue
        output += f'_{key}_{log_list[key]}'
    
    if 'etc' in log_list.keys():
        if log_list['etc'] != None:
            output = output + '_' + log_list['etc']

    print_log(f'Log name: \n\t{output}')
    return output
