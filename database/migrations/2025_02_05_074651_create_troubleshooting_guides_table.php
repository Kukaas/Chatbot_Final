<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    public function up()
    {
        Schema::create('troubleshooting_guides', function (Blueprint $table) {
            $table->id();
            $table->string('title');
            $table->text('content');
            $table->json('embedding')->nullable();
            $table->boolean('is_active')->default(true);
            $table->timestamps();
        });
    }

    public function down()
    {
        Schema::dropIfExists('troubleshooting_guides');
    }
};