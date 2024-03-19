defmodule Mighty.Utils do
  def ngram_range(tokens, {min_n, max_n}) do
    n_original_tokens = length(tokens)

    ngrams =
      for n <- min_n..min(max_n, n_original_tokens),
          i <- 0..(n_original_tokens - n) do
        Enum.slice(tokens, i, n) |> Enum.join(" ")
      end

    ngrams
  end

  def unigrams(tokens) do
    ngram_range(tokens, {1, 1})
  end

  def bigrams(tokens) do
    ngram_range(tokens, {2, 2})
  end

  def trigrams(tokens) do
    ngram_range(tokens, {3, 3})
  end

  def ngrams(tokens, n) do
    ngram_range(tokens, {n, n})
  end

  def invert_map(map) do
    map
    |> Enum.map(fn {k, v} -> {v, k} end)
    |> Map.new()
  end

  @doc """
  Pads a sequence to a given length.

  ## Examples
  ```elixir
  iex> alias Mighty.Utils, as: U
  iex> U.pad_sequence([1,2,3,4,5], 2, pad_left: true, pad_right: true, left_pad_symbol: '<s>', right_pad_symbol: '</s>')
  ['<s>', 1, 2, 3, 4, 5, '</s>']
  iex> U.pad_sequence([1,2,3,4,5], 2, pad_left: true, left_pad_symbol: '<s>')
  ['<s>', 1, 2, 3, 4, 5]
  iex> U.pad_sequence([1,2,3,4,5], 2, pad_right: true, right_pad_symbol: '</s>')
  [1, 2, 3, 4, 5, '</s>']
  ```
  """
  def pad_sequence(
        sequence,
        n,
        opts \\ []
      ) do
    opts =
      Keyword.validate!(opts,
        pad_left: false,
        pad_right: false,
        left_pad_symbol: nil,
        right_pad_symbol: nil
      )

    n = n - 1
    pad_left = opts[:pad_left]
    pad_right = opts[:pad_right]
    left_pad_symbol = opts[:left_pad_symbol]
    right_pad_symbol = opts[:right_pad_symbol]

    case {pad_left, pad_right} do
      {true, true} ->
        List.duplicate(left_pad_symbol, n) ++ sequence ++ List.duplicate(right_pad_symbol, n)

      {true, false} ->
        List.duplicate(left_pad_symbol, n) ++ sequence

      {false, true} ->
        sequence ++ List.duplicate(right_pad_symbol, n)

      {false, false} ->
        sequence
    end
  end
end
