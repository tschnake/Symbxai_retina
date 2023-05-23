import torch


def load_pretrained_weights(
        model,
        path
):
    params = torch.load(
        path,
        map_location=torch.device('cpu')
    )

    state_dict = model.state_dict()

    for i in range(3):
        state_dict['bert.encoder.layer.{}.attention.self.query.weight'.format(i)] = params[
            'attention_layers.{}.query.weight'.format(i)]
        state_dict['bert.encoder.layer.{}.attention.self.key.weight'.format(i)] = params[
            'attention_layers.{}.key.weight'.format(i)]
        state_dict['bert.encoder.layer.{}.attention.self.value.weight'.format(i)] = params[
            'attention_layers.{}.value.weight'.format(i)]

        state_dict['bert.encoder.layer.{}.attention.self.query.bias'.format(i)] = params[
            'attention_layers.{}.query.bias'.format(i)]
        state_dict['bert.encoder.layer.{}.attention.self.key.bias'.format(i)] = params[
            'attention_layers.{}.key.bias'.format(i)]
        state_dict['bert.encoder.layer.{}.attention.self.value.bias'.format(i)] = params[
            'attention_layers.{}.value.bias'.format(i)]

        state_dict['bert.encoder.layer.{}.attention.output.dense.weight'.format(i)] = params[
            'attention_layers.{}.output.dense.weight'.format(i)]
        state_dict['bert.encoder.layer.{}.attention.output.dense.bias'.format(i)] = params[
            'attention_layers.{}.output.dense.bias'.format(i)]

        state_dict['bert.encoder.layer.{}.attention.output.LayerNorm.weight'.format(i)] = params[
            'attention_layers.{}.output.LayerNorm.weight'.format(i)]
        state_dict['bert.encoder.layer.{}.attention.output.LayerNorm.bias'.format(i)] = params[
            'attention_layers.{}.output.LayerNorm.bias'.format(i)]

        state_dict['bert.pooler.dense.weight'] = params['pooler.dense.weight']
        state_dict['bert.pooler.dense.bias'] = params['pooler.dense.bias']

        state_dict['classifier.weight'] = params['classifier.weight']
        state_dict['classifier.bias'] = params['classifier.bias']

    model.load_state_dict(state_dict)
