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
    state_dict['bert.embeddings.position_embeddings.weight'] = params[
        'embeds.position_embeddings.weight']
    state_dict['bert.embeddings.token_type_embeddings.weight'] = params[
        'embeds.token_type_embeddings.weight']
    state_dict['bert.embeddings.word_embeddings.weight'] = params[
        'embeds.word_embeddings.weight']
    state_dict['bert.embeddings.position_ids'] = params[
        'embeds.position_ids']

    state_dict['bert.embeddings.LayerNorm.weight'] = params[
        'embeds.LayerNorm.weight']
    state_dict['bert.embeddings.LayerNorm.bias'] = params[
        'embeds.LayerNorm.bias']


    for i in range(3):
        state_dict['bert.encoder.layer.{}.self.query.weight'.format(i)] = params[
            'attention_layers.{}.query.weight'.format(i)]
        state_dict['bert.encoder.layer.{}.self.key.weight'.format(i)] = params[
            'attention_layers.{}.key.weight'.format(i)]
        state_dict['bert.encoder.layer.{}.self.value.weight'.format(i)] = params[
            'attention_layers.{}.value.weight'.format(i)]

        state_dict['bert.encoder.layer.{}.self.query.bias'.format(i)] = params[
            'attention_layers.{}.query.bias'.format(i)]
        state_dict['bert.encoder.layer.{}.self.key.bias'.format(i)] = params[
            'attention_layers.{}.key.bias'.format(i)]
        state_dict['bert.encoder.layer.{}.self.value.bias'.format(i)] = params[
            'attention_layers.{}.value.bias'.format(i)]

        state_dict['bert.encoder.layer.{}.output.dense.weight'.format(i)] = params[
            'attention_layers.{}.output.dense.weight'.format(i)]
        state_dict['bert.encoder.layer.{}.output.dense.bias'.format(i)] = params[
            'attention_layers.{}.output.dense.bias'.format(i)]

        state_dict['bert.encoder.layer.{}.output.LayerNorm.weight'.format(i)] = params[
            'attention_layers.{}.output.LayerNorm.weight'.format(i)]
        state_dict['bert.encoder.layer.{}.output.LayerNorm.bias'.format(i)] = params[
            'attention_layers.{}.output.LayerNorm.bias'.format(i)]

        state_dict['bert.pooler.dense.weight'] = params['pooler.dense.weight']
        state_dict['bert.pooler.dense.bias'] = params['pooler.dense.bias']

        state_dict['classifier.weight'] = params['classifier.weight']
        state_dict['classifier.bias'] = params['classifier.bias']

    model.load_state_dict(state_dict)
